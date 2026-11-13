//! Moving learning data between devices and the home device.
//!
//! One "home" profile (on the desktop) owns the single trainable adapter
//! shared by every project and device. Devices run the latest version they
//! have received; their feedback is shipped home during an authenticated
//! sync, trained into the next version, and delivered on a later sync.
//!
//! All payloads are opaque byte blobs; the transport (Portal) doesn't parse them.
//!
//! Per sync, device ↔ home:
//!   device: b = exportBundle(max_bytes)            -> send b.bytes + profile.version
//!   home:   importBundle(bytes), durably stored    -> ack
//!   device: commitExport(b.through)                (repeat while more is pending)
//!   home:   exportUpdate(device_version)           -> send if non-null
//!   device: importUpdate(bytes)                    (used on the next model load)
//!   home:   training runs on its schedule (see home.zig / schedule.zig)
const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const profile_mod = @import("profile.zig");
const Profile = profile_mod.Profile;
const audio_mod = @import("audio.zig");

const bundle_magic = "STTBUN01";
const update_magic = "STTUPD01";

/// Largest single sample in a bundle: 30 s of 16-bit audio plus text.
pub const max_sample_bytes = profile_mod.max_sample_seconds * audio_mod.target_rate * 2 + 64 * 1024;
/// Cap on a sample's text fields (transcript, heard text, metadata, device name).
pub const max_text_bytes = 16 * 1024;
/// Suggested per-request cap for bundles (several samples, chunked).
pub const default_bundle_bytes = 8 << 20;

pub const Bundle = struct {
    bytes: []u8,
    /// Pass to `commitExport` once the home device has stored the bundle.
    through: u32,
    count: u32,

    pub fn deinit(b: *Bundle, gpa: Allocator) void {
        gpa.free(b.bytes);
    }
};

// ------------------------------------------------------------------ device

/// Packs pending samples (oldest first) into a bundle of at most `max_bytes`
/// (always at least one sample). Returns null if nothing is pending or
/// learning is off. Call repeatedly (after each commit) to drain.
pub fn exportBundle(gpa: Allocator, p: *Profile, device_name: []const u8, max_bytes: usize) !?Bundle {
    if (!p.learning) return null;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, bundle_magic);
    try putInt(gpa, &out, p.version);
    try putBytes(gpa, &out, device_name);
    const count_at = out.items.len;
    try putInt(gpa, &out, 0);

    var count: u32 = 0;
    var through = p.exported;
    var id = p.exported + 1;
    var name_buf: [48]u8 = undefined;
    while (id < p.next_sample) : (id += 1) {
        const text = readSampleFile(gpa, p, &name_buf, id, "txt") orelse continue;
        defer gpa.free(text);
        const audio = readSampleFile(gpa, p, &name_buf, id, "wav") orelse continue;
        defer gpa.free(audio);
        const meta = readSampleFile(gpa, p, &name_buf, id, "meta") orelse try gpa.dupe(u8, "");
        defer gpa.free(meta);
        const entry_len = 4 * 3 + text.len + audio.len + meta.len;
        if (count > 0 and out.items.len + entry_len > max_bytes) break;
        try putBytes(gpa, &out, meta);
        try putBytes(gpa, &out, text);
        try putBytes(gpa, &out, audio);
        count += 1;
        through = id;
    }
    // Skipped (incomplete) ids up to the scan position are done too.
    if (count == 0) {
        if (id > p.exported + 1) {
            p.exported = id - 1;
            try p.saveState();
        }
        out.deinit(gpa);
        return null;
    }
    std.mem.writeInt(u32, out.items[count_at..][0..4], count, .little);
    return .{ .bytes = try out.toOwnedSlice(gpa), .through = through, .count = count };
}

/// Marks samples up to `through` as delivered and deletes them from this
/// device (the home device now holds them).
pub fn commitExport(p: *Profile, through: u32) !void {
    if (through <= p.exported) return;
    var name_buf: [48]u8 = undefined;
    var id = p.exported + 1;
    while (id <= through) : (id += 1) {
        for ([_][]const u8{ "wav", "meta", "txt" }) |ext| {
            const rel = std.fmt.bufPrint(&name_buf, "samples/{d:0>6}.{s}", .{ id, ext }) catch unreachable;
            Io.Dir.cwd().deleteFile(p.io, try p.path(rel)) catch {};
        }
    }
    p.exported = through;
    try p.saveState();
}

/// Installs an update from the home device: adapter weights plus phrase
/// rules and vocabulary. Takes effect the next time a model is loaded with
/// this profile. Returns the version now installed.
pub fn importUpdate(gpa: Allocator, p: *Profile, bytes: []const u8) !u32 {
    var r: Reader = .{ .buf = bytes };
    if (!std.mem.eql(u8, try r.take(8), update_magic)) return error.BadUpdate;
    const version = try r.int();
    const adapter = try r.bytes();
    const rules = try r.bytes();
    const vocab = try r.bytes();
    if (version <= p.version) return p.version;

    const cwd = Io.Dir.cwd();
    if (adapter.len > 0) {
        try cwd.writeFile(p.io, .{ .sub_path = try p.path("adapter.bin.tmp"), .data = adapter });
        try cwd.rename(try p.path("adapter.bin.tmp"), cwd, try p.path("adapter.bin"), p.io);
    }
    try p.mergeRulesText(rules);
    try p.mergeVocabText(vocab);
    try p.save(gpa);
    p.version = version;
    try p.saveState();
    return version;
}

// ------------------------------------------------------------------ home

pub const ImportReport = struct {
    samples: u32 = 0,
    fixes: u32 = 0,
    /// Adapter version the device was running when it produced these.
    device_version: u32 = 0,
};

/// Stores a device's bundle in the home profile; fixes also teach phrase
/// rules and vocabulary right away. Rejects malformed input.
pub fn importBundle(gpa: Allocator, home: *Profile, bytes: []const u8) !ImportReport {
    var r: Reader = .{ .buf = bytes };
    if (!std.mem.eql(u8, try r.take(8), bundle_magic)) return error.BadBundle;
    var rep: ImportReport = .{ .device_version = try r.int() };
    const device = try r.bytes();
    if (device.len > max_text_bytes) return error.BadBundle;
    const count = try r.int();
    const body = r;

    // Pass 1: check the whole bundle's structure and sizes before storing
    // anything, so a bad entry can't leave a half-imported bundle behind
    // (the device would resend it and home would get duplicates).
    for (0..count) |_| {
        const meta_text = try r.bytes();
        const text = try r.bytes();
        const audio_bytes = try r.bytes();
        if (audio_bytes.len > max_sample_bytes) return error.BadBundle;
        if (meta_text.len > max_text_bytes or text.len > max_text_bytes) return error.BadBundle;
        if (parseMeta(meta_text).heard.len > max_text_bytes) return error.BadBundle;
    }

    // Pass 2: store.
    r = body;
    for (0..count) |_| {
        const meta_text = try r.bytes();
        const text = try r.bytes();
        const audio_bytes = try r.bytes();
        const samples = audio_mod.decode(gpa, audio_bytes, .{ .max_seconds = profile_mod.max_sample_seconds + 1 }) catch return error.BadBundle;
        defer gpa.free(samples);
        const meta = parseMeta(meta_text);
        const source = if (meta.source.len > 0) meta.source else device;
        if (meta.kind == .fix and meta.heard.len > 0) {
            _ = try home.learnFrom(gpa, meta.heard, text, samples, source);
            rep.fixes += 1;
            rep.samples += 1;
        } else if (try home.saveSampleMeta(gpa, samples, text, .{ .kind = .confirm, .heard = text, .source = source })) |_| {
            rep.samples += 1;
        }
    }
    try home.save(gpa);
    return rep;
}

pub const StoragePolicy = struct {
    /// When the home profile's samples exceed this, compact them.
    max_bytes: u64 = 50 << 30,
    /// Compact down to this fraction of `max_bytes`.
    target_ratio: f64 = 0.8,
    /// Keep the audio of every Nth sample as a replay set, so later
    /// versions don't forget older speech (0 = keep none).
    keep_every: u32 = 10,
};

pub const CompactReport = struct {
    freed_bytes: u64 = 0,
    deleted_samples: u32 = 0,
    bytes_after: u64 = 0,
};

/// Builds the update for a device running `device_version`, or null if it
/// is already current.
pub fn exportUpdate(gpa: Allocator, home: *Profile, device_version: u32) !?[]u8 {
    if (home.version <= device_version) return null;
    const cwd = Io.Dir.cwd();
    const adapter = cwd.readFileAlloc(home.io, try home.path("adapter.bin"), gpa, .limited(1 << 30)) catch |e| switch (e) {
        error.FileNotFound => try gpa.dupe(u8, ""),
        else => return e,
    };
    defer gpa.free(adapter);
    const rules = cwd.readFileAlloc(home.io, try home.path("rules.txt"), gpa, .limited(16 << 20)) catch try gpa.dupe(u8, "");
    defer gpa.free(rules);
    const vocab = cwd.readFileAlloc(home.io, try home.path("vocab.txt"), gpa, .limited(16 << 20)) catch try gpa.dupe(u8, "");
    defer gpa.free(vocab);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, update_magic);
    try putInt(gpa, &out, home.version);
    try putBytes(gpa, &out, adapter);
    try putBytes(gpa, &out, rules);
    try putBytes(gpa, &out, vocab);
    return try out.toOwnedSlice(gpa);
}

/// Deletes the audio of the oldest samples that are already trained into the
/// adapter until usage is under `target_ratio * max_bytes`, sparing every
/// `keep_every`-th sample as a replay set. Text and metadata are kept.
/// Untrained samples are never deleted.
pub fn compact(home: *Profile, policy: StoragePolicy) !CompactReport {
    var rep: CompactReport = .{};
    var size = try samplesBytes(home);
    const target: u64 = @intFromFloat(@as(f64, @floatFromInt(policy.max_bytes)) * policy.target_ratio);
    var name_buf: [48]u8 = undefined;
    var id: u32 = 1;
    while (size > target and id <= home.trained) : (id += 1) {
        if (policy.keep_every > 0 and id % policy.keep_every == 0) continue;
        const rel = std.fmt.bufPrint(&name_buf, "samples/{d:0>6}.wav", .{id}) catch unreachable;
        const full = try home.path(rel);
        const st = Io.Dir.cwd().statFile(home.io, full, .{}) catch continue;
        Io.Dir.cwd().deleteFile(home.io, full) catch continue;
        size -|= st.size;
        rep.freed_bytes += st.size;
        rep.deleted_samples += 1;
    }
    rep.bytes_after = size;
    return rep;
}

/// Total bytes of sample files in the profile.
pub fn samplesBytes(p: *Profile) !u64 {
    var dir = try Io.Dir.cwd().openDir(p.io, try p.path("samples"), .{ .iterate = true });
    defer dir.close(p.io);
    var total: u64 = 0;
    var it = dir.iterate();
    while (try it.next(p.io)) |e| {
        if (e.kind != .file) continue;
        const st = dir.statFile(p.io, e.name, .{}) catch continue;
        total += st.size;
    }
    return total;
}

// ------------------------------------------------------------------ helpers

fn readSampleFile(gpa: Allocator, p: *Profile, buf: *[48]u8, id: u32, ext: []const u8) ?[]u8 {
    const rel = std.fmt.bufPrint(buf, "samples/{d:0>6}.{s}", .{ id, ext }) catch return null;
    const full = p.path(rel) catch return null;
    return Io.Dir.cwd().readFileAlloc(p.io, full, gpa, .limited(max_sample_bytes)) catch null;
}

fn parseMeta(text: []const u8) profile_mod.SampleMeta {
    var m: profile_mod.SampleMeta = .{};
    var lines = std.mem.tokenizeAny(u8, text, "\r\n");
    while (lines.next()) |line| {
        const eq = std.mem.indexOfScalar(u8, line, '=') orelse continue;
        const k = line[0..eq];
        const v = line[eq + 1 ..];
        if (std.mem.eql(u8, k, "kind")) m.kind = std.meta.stringToEnum(profile_mod.SampleKind, v) orelse .confirm;
        if (std.mem.eql(u8, k, "heard")) m.heard = v;
        if (std.mem.eql(u8, k, "source")) m.source = v;
    }
    return m;
}

fn putInt(gpa: Allocator, out: *std.ArrayList(u8), v: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try out.appendSlice(gpa, &b);
}

fn putBytes(gpa: Allocator, out: *std.ArrayList(u8), data: []const u8) !void {
    try putInt(gpa, out, @intCast(data.len));
    try out.appendSlice(gpa, data);
}

/// Bounds-checked reader for untrusted payloads.
const Reader = struct {
    buf: []const u8,
    pos: usize = 0,

    fn take(r: *Reader, n: usize) ![]const u8 {
        if (r.buf.len - r.pos < n) return error.Truncated;
        defer r.pos += n;
        return r.buf[r.pos..][0..n];
    }
    fn int(r: *Reader) !u32 {
        return std.mem.readInt(u32, (try r.take(4))[0..4], .little);
    }
    fn bytes(r: *Reader) ![]const u8 {
        return r.take(try r.int());
    }
};

test "device -> home -> device round trip" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const ddir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/device", .{tmp.sub_path});
    defer gpa.free(ddir);
    const hdir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/home", .{tmp.sub_path});
    defer gpa.free(hdir);

    var dev = try Profile.load(gpa, io, ddir);
    defer dev.deinit();
    var home = try Profile.load(gpa, io, hdir);
    defer home.deinit();

    const audio: [3200]f32 = @splat(0.01);
    _ = try dev.feedback(gpa, &audio, "Hello world.", .{ .text = "Hello world.", .success = true });
    _ = try dev.feedback(gpa, &audio, "written in Puresig.", .{ .text = "written in pure Zig.", .success = true });
    _ = try dev.feedback(gpa, &audio, "Learning's egg.", .{ .text = "", .success = false }); // discarded

    // Chunking: a tiny cap still makes progress one sample at a time.
    var b1 = (try exportBundle(gpa, &dev, "laptop", 1)).?;
    defer b1.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 1), b1.count);
    const r1 = try importBundle(gpa, &home, b1.bytes);
    try std.testing.expectEqual(@as(u32, 0), r1.fixes);
    try commitExport(&dev, b1.through);

    var b2 = (try exportBundle(gpa, &dev, "laptop", default_bundle_bytes)).?;
    defer b2.deinit(gpa);
    const r2 = try importBundle(gpa, &home, b2.bytes);
    try std.testing.expectEqual(@as(u32, 1), r2.fixes);
    try commitExport(&dev, b2.through);
    try std.testing.expect((try exportBundle(gpa, &dev, "laptop", default_bundle_bytes)) == null);

    // Home learned the rule from the device's fix.
    const fixed = try home.applyRules(gpa, "made in Puresig");
    defer gpa.free(fixed);
    try std.testing.expectEqualStrings("made in pure Zig", fixed);
    try std.testing.expectEqual(@as(u32, 3), home.next_sample);

    // Pretend home trained version 1; device picks up the rules.
    home.version = 1;
    var fresh = try Profile.load(gpa, io, try std.fmt.allocPrint(home.arena.allocator(), "{s}/../device2", .{hdir}));
    defer fresh.deinit();
    const up = (try exportUpdate(gpa, &home, fresh.version)).?;
    defer gpa.free(up);
    try std.testing.expectEqual(@as(u32, 1), try importUpdate(gpa, &fresh, up));
    try std.testing.expect((try exportUpdate(gpa, &home, fresh.version)) == null);
    const fixed2 = try fresh.applyRules(gpa, "made in Puresig");
    defer gpa.free(fixed2);
    try std.testing.expectEqualStrings("made in pure Zig", fixed2);

    // Compaction: with a tiny cap, trained audio goes except the replay set.
    for (0..9) |_| _ = try home.saveSample(gpa, &audio, "filler");
    home.trained = home.next_sample - 1; // pretend all are trained
    const before = try samplesBytes(&home);
    const rep = try compact(&home, .{ .max_bytes = 1, .target_ratio = 0, .keep_every = 5 });
    try std.testing.expect(rep.deleted_samples > 0 and rep.bytes_after < before);
    var nb: [48]u8 = undefined;
    const kept = try home.path(try std.fmt.bufPrint(&nb, "samples/{d:0>6}.wav", .{5}));
    _ = try Io.Dir.cwd().statFile(io, kept, .{}); // replay sample survives
    const gone = try home.path(try std.fmt.bufPrint(&nb, "samples/{d:0>6}.wav", .{1}));
    try std.testing.expectError(error.FileNotFound, Io.Dir.cwd().statFile(io, gone, .{}));

    // Malformed input is rejected, not trusted.
    try std.testing.expectError(error.BadBundle, importBundle(gpa, &home, "NOTABUNDLE......"));
    try std.testing.expectError(error.Truncated, importBundle(gpa, &home, "nope"));
    try std.testing.expectError(error.Truncated, importBundle(gpa, &home, bundle_magic ++ "\x01\x00"));
}
