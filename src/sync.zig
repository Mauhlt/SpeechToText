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
//!   home:   importBundle(bytes, device), durably stored -> ack
//!   device: commitExport(b.through)                (repeat while more is pending)
//!   home:   exportUpdate(device_version)           -> send if non-null
//!   device: importUpdate(bytes)                    (used on the next model load)
//!   home:   training runs on its schedule (see home.zig / schedule.zig)
//!
//! Home never trusts a bundle's own device name: the caller passes the
//! authenticated device it came from, and everything it teaches is recorded
//! under that name (`Profile.forgetSource` undoes it).
const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const profile_mod = @import("profile.zig");
const Profile = profile_mod.Profile;
const audio_mod = @import("audio.zig");
const Model = @import("model.zig").Model;
const Adapter = @import("adapter.zig").Adapter;

const bundle_magic = "STTBUN01";
const update_magic = "STTUPD01";

/// Largest single sample in a bundle: 30 s of 16-bit audio plus text.
pub const max_sample_bytes = profile_mod.max_sample_seconds * audio_mod.target_rate * 2 + 64 * 1024;
/// Cap on a sample's text fields (transcript, heard text, metadata, device name).
pub const max_text_bytes = 16 * 1024;
/// Suggested per-request cap for bundles (several samples, chunked).
pub const default_bundle_bytes = 8 << 20;
/// Most samples one bundle may carry (`exportBundle` stops here).
pub const max_bundle_entries = 1024;
/// Most new rules and words one bundle may add to the home profile.
pub const max_bundle_new_rules = 64;
pub const max_bundle_new_words = 256;

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
    while (id < p.next_sample and count < max_bundle_entries) : (id += 1) {
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
    // Only ids that exist can have been exported (this also keeps the loop
    // finite for through = maxInt(u32)).
    const last = @min(through, p.next_sample -| 1);
    if (last <= p.exported) return;
    var name_buf: [48]u8 = undefined;
    var id = p.exported + 1;
    while (id <= last) : (id += 1) {
        for ([_][]const u8{ "txt", "wav", "meta" }) |ext| {
            const rel = std.fmt.bufPrint(&name_buf, "samples/{d:0>6}.{s}", .{ id, ext }) catch unreachable;
            Io.Dir.cwd().deleteFile(p.io, try p.path(rel)) catch {};
        }
    }
    p.exported = last;
    try p.saveState();
}

/// Installs an update from the home device: adapter weights plus phrase
/// rules and vocabulary. Takes effect the next time a model is loaded with
/// this profile. Returns the version now installed. Without the model the
/// adapter is checked for internal consistency only; `importUpdateFor`
/// also checks it fits the model.
pub fn importUpdate(gpa: Allocator, p: *Profile, bytes: []const u8) !u32 {
    return importUpdateFor(gpa, p, bytes, null);
}

/// Like `importUpdate`, checking the adapter against `model` (layer count,
/// width, size) before anything is written. A bad update changes nothing:
/// the adapter is validated first, and the version is recorded last.
pub fn importUpdateFor(gpa: Allocator, p: *Profile, bytes: []const u8, model: ?*const Model) !u32 {
    var r: Reader = .{ .buf = bytes };
    if (!std.mem.eql(u8, try r.take(8), update_magic)) return error.BadUpdate;
    const version = try r.int();
    const adapter = try r.bytes();
    const rules = try r.bytes();
    const vocab = try r.bytes();
    if (version <= p.version) return p.version;

    if (adapter.len > 0) {
        if (model) |m| {
            var ad = try Adapter.parse(gpa, adapter, m);
            ad.deinit(gpa);
        } else try Adapter.check(adapter);
    }
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

/// Stores a bundle from `source_device` in the home profile; fixes also
/// teach phrase rules and vocabulary right away. `source_device` is the
/// authenticated device the caller received it from (the bundle's own
/// device name is ignored); it is recorded on every sample, rule and word.
/// All or nothing for bad input: the whole bundle is checked and decoded
/// before anything is stored. See `prepareBundle` / `commitBundle` to do
/// the expensive half without holding the profile's lock.
pub fn importBundle(gpa: Allocator, home: *Profile, bytes: []const u8, source_device: []const u8) !ImportReport {
    var pb = try prepareBundle(gpa, bytes, source_device);
    defer pb.deinit();
    return commitBundle(gpa, home, &pb);
}

/// A bundle checked, decoded and turned into lessons, ready to store.
pub const PreparedBundle = struct {
    arena: *std.heap.ArenaAllocator,
    source: []const u8,
    device_version: u32,
    entries: []const Entry,

    pub const Entry = struct {
        kind: profile_mod.SampleKind,
        text: []const u8,
        heard: []const u8,
        /// 16 kHz mono WAV to store, or null (too long or empty: not kept).
        wav: ?[]const u8,
        lesson: profile_mod.Lesson,
    };

    pub fn deinit(pb: *PreparedBundle) void {
        const gpa = pb.arena.child_allocator;
        pb.arena.deinit();
        gpa.destroy(pb.arena);
        pb.* = undefined;
    }
};

/// The expensive, profile-free half of `importBundle` (safe outside any
/// lock): checks every size, decodes every sample's audio, cleans control
/// characters out of the texts and works out what each fix teaches (with
/// the word-count and word-length caps). Any bad entry rejects the bundle.
pub fn prepareBundle(gpa: Allocator, bytes: []const u8, source_device: []const u8) !PreparedBundle {
    try profile_mod.checkSource(source_device);
    const arena = try gpa.create(std.heap.ArenaAllocator);
    errdefer gpa.destroy(arena);
    arena.* = .init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    var r: Reader = .{ .buf = bytes };
    if (!std.mem.eql(u8, try r.take(8), bundle_magic)) return error.BadBundle;
    const device_version = try r.int();
    // The bundle's own device name: checked for size, otherwise unused.
    if ((try r.bytes()).len > max_text_bytes) return error.BadBundle;
    const count = try r.int();
    if (count > max_bundle_entries) return error.BadBundle;
    const entries = try a.alloc(PreparedBundle.Entry, count);
    for (entries) |*e| {
        const meta_text = try r.bytes();
        const text_raw = try r.bytes();
        const audio_bytes = try r.bytes();
        if (audio_bytes.len > max_sample_bytes) return error.BadBundle;
        if (meta_text.len > max_text_bytes or text_raw.len > max_text_bytes) return error.BadBundle;
        const meta = profile_mod.SampleMeta.parse(meta_text);
        const text = try profile_mod.cleanText(a, text_raw);
        const heard = try profile_mod.cleanText(a, meta.heard);
        const samples = audio_mod.decode(gpa, audio_bytes, .{ .max_seconds = profile_mod.max_sample_seconds + 1 }) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => return error.BadBundle,
        };
        defer gpa.free(samples);
        const keep = text.len > 0 and samples.len <= profile_mod.max_sample_seconds * audio_mod.target_rate;
        const wav: ?[]const u8 = if (keep) blk: {
            const w = try audio_mod.encodeWav(gpa, samples, audio_mod.target_rate);
            defer gpa.free(w);
            break :blk try a.dupe(u8, w);
        } else null;
        const fix = meta.kind == .fix and heard.len > 0 and text.len > 0 and !std.mem.eql(u8, heard, text);
        e.* = .{
            .kind = if (fix) .fix else .confirm,
            .text = text,
            .heard = if (fix) heard else text,
            .wav = wav,
            .lesson = if (fix) try profile_mod.lesson(a, heard, text) else .{},
        };
    }
    if (r.pos != r.buf.len) return error.BadBundle;
    return .{
        .arena = arena,
        .source = try a.dupe(u8, source_device),
        .device_version = device_version,
        .entries = entries,
    };
}

/// The cheap half of `importBundle` (caller holds the profile's lock):
/// stores the samples, applies the lessons within the per-bundle and total
/// caps, and writes vocab.txt / rules.txt / sources.txt once.
pub fn commitBundle(gpa: Allocator, home: *Profile, pb: *const PreparedBundle) !ImportReport {
    var rep: ImportReport = .{ .device_version = pb.device_version };
    var budget: profile_mod.Budget = .{ .rules = max_bundle_new_rules, .words = max_bundle_new_words };
    for (pb.entries) |e| {
        if (e.kind == .fix) {
            _ = try home.applyLesson(e.lesson, pb.source, &budget);
            rep.fixes += 1;
        }
        if (e.wav) |wav| {
            _ = try home.writeSample(wav, e.text, .{ .kind = e.kind, .heard = e.heard, .source = pb.source });
            rep.samples += 1;
        }
    }
    home.sortRules();
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

pub const CompactReport = profile_mod.StorageReport;

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
    const r1 = try importBundle(gpa, &home, b1.bytes, "laptop");
    try std.testing.expectEqual(@as(u32, 0), r1.fixes);
    try commitExport(&dev, b1.through);

    var b2 = (try exportBundle(gpa, &dev, "laptop", default_bundle_bytes)).?;
    defer b2.deinit(gpa);
    // The bundle claims "laptop"; the session says it came from "tablet".
    const r2 = try importBundle(gpa, &home, b2.bytes, "tablet");
    try std.testing.expectEqual(@as(u32, 1), r2.fixes);
    try std.testing.expectEqualStrings("tablet", home.rules.items[0].source);
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
    for (0..9) |_| _ = try profile_mod.testSample(&home, gpa, &audio, "filler");
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
    try std.testing.expectError(error.BadBundle, importBundle(gpa, &home, "NOTABUNDLE......", "laptop"));
    try std.testing.expectError(error.Truncated, importBundle(gpa, &home, "nope", "laptop"));
    try std.testing.expectError(error.Truncated, importBundle(gpa, &home, bundle_magic ++ "\x01\x00", "laptop"));
    try std.testing.expectError(error.InvalidSource, importBundle(gpa, &home, b2.bytes, ""));
    try std.testing.expectError(error.InvalidSource, importBundle(gpa, &home, b2.bytes, "\r\n"));
}

/// A bundle built by hand: (meta, text, audio) per entry.
fn testBundle(gpa: Allocator, entries: []const [3][]const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, bundle_magic);
    try putInt(gpa, &out, 0);
    try putBytes(gpa, &out, "claimed-name");
    try putInt(gpa, &out, @intCast(entries.len));
    for (entries) |e| for (e) |f| try putBytes(gpa, &out, f);
    return out.toOwnedSlice(gpa);
}

test "importBundle: a bad entry stores nothing; control characters are cleaned; caps hold" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const hdir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/home", .{tmp.sub_path});
    defer gpa.free(hdir);
    var home = try Profile.load(gpa, io, hdir);
    defer home.deinit();
    const tone: [1600]f32 = @splat(0.01);
    const wav = try audio_mod.encodeWav(gpa, &tone, audio_mod.target_rate);
    defer gpa.free(wav);

    // S4: the last entry's audio is garbage, so the good fix before it
    // must not be stored or learned either.
    const bad = try testBundle(gpa, &.{
        .{ "kind=fix\nheard=written in Puresig.\n", "written in pure Zig.", wav },
        .{ "kind=confirm\n", "fine", "not audio at all" },
    });
    defer gpa.free(bad);
    try std.testing.expectError(error.BadBundle, importBundle(gpa, &home, bad, "laptop"));
    try std.testing.expectEqual(@as(u32, 1), home.next_sample);
    try std.testing.expectEqual(@as(usize, 0), home.rules.items.len);
    try std.testing.expectEqual(@as(usize, 0), home.words.count());
    // Trailing bytes and too many entries are rejected too.
    const one = try testBundle(gpa, &.{.{ "kind=confirm\n", "fine", wav }});
    defer gpa.free(one);
    const trailing = try std.mem.concat(gpa, u8, &.{ one, "x" });
    defer gpa.free(trailing);
    try std.testing.expectError(error.BadBundle, importBundle(gpa, &home, trailing, "laptop"));
    var many: [12]u8 = undefined;
    @memcpy(many[0..8], bundle_magic);
    std.mem.writeInt(u32, many[8..12], 0, .little);
    const huge = try std.mem.concat(gpa, u8, &.{ &many, "\x00\x00\x00\x00", &[4]u8{ 0x01, 0x04, 0, 0 } });
    defer gpa.free(huge);
    try std.testing.expectError(error.BadBundle, importBundle(gpa, &home, huge, "laptop"));

    // S3: control characters never reach the files; the session's name is the source.
    const ctl = try testBundle(gpa, &.{.{ "kind=fix\nsource=evil\nheard=meet Bobbie\x1b[2J today\n", "meet Bobby\x07 today\n", wav }});
    defer gpa.free(ctl);
    const rep = try importBundle(gpa, &home, ctl, "phone");
    try std.testing.expectEqual(@as(u32, 1), rep.samples);
    try std.testing.expectEqual(@as(u32, 1), rep.fixes);
    for (home.rules.items) |r| {
        for (r.to) |c| try std.testing.expect(c >= 0x20 and c != 0x7f);
        try std.testing.expectEqualStrings("phone", r.source);
    }
    var nb: [48]u8 = undefined;
    const meta = try Io.Dir.cwd().readFileAlloc(io, try home.path(try std.fmt.bufPrint(&nb, "samples/{d:0>6}.meta", .{1})), gpa, .limited(4096));
    defer gpa.free(meta);
    try std.testing.expect(std.mem.indexOf(u8, meta, "source=phone\n") != null);
    try std.testing.expect(std.mem.indexOfScalar(u8, meta, 0x1b) == null);
    const txt = try Io.Dir.cwd().readFileAlloc(io, try home.path(try std.fmt.bufPrint(&nb, "samples/{d:0>6}.txt", .{1})), gpa, .limited(4096));
    defer gpa.free(txt);
    try std.testing.expectEqualStrings("meet Bobby  today", txt);

    // S2: one bundle teaches at most max_bundle_new_rules rules.
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var list: std.ArrayList([3][]const u8) = .empty;
    for (0..max_bundle_new_rules + 20) |i| {
        const m = try std.fmt.allocPrint(a, "kind=fix\nheard=say wrongword{d} now\n", .{i});
        const t = try std.fmt.allocPrint(a, "say Right{d} now", .{i});
        try list.append(a, .{ m, t, wav });
    }
    const big = try testBundle(gpa, list.items);
    defer gpa.free(big);
    const before = home.rules.items.len;
    _ = try importBundle(gpa, &home, big, "phone");
    try std.testing.expectEqual(before + max_bundle_new_rules, home.rules.items.len);
}

test "commitExport: through = maxInt(u32) ends and records only real ids" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    var dev = try Profile.load(gpa, io, dir);
    defer dev.deinit();
    const audio: [800]f32 = @splat(0);
    _ = try profile_mod.testSample(&dev, gpa, &audio, "one");
    try commitExport(&dev, std.math.maxInt(u32));
    try std.testing.expectEqual(@as(u32, 1), dev.exported);
    var again = try Profile.load(gpa, io, dir);
    defer again.deinit();
    try std.testing.expectEqual(@as(u32, 2), again.next_sample);
}

test "importUpdate: a bad adapter changes nothing" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    var dev = try Profile.load(gpa, io, dir);
    defer dev.deinit();

    const build = struct {
        fn update(g: Allocator, adapter: []const u8) ![]u8 {
            var out: std.ArrayList(u8) = .empty;
            errdefer out.deinit(g);
            try out.appendSlice(g, update_magic);
            try putInt(g, &out, 3);
            try putBytes(g, &out, adapter);
            try putBytes(g, &out, "puresig => pure Zig\n");
            try putBytes(g, &out, "5\tZig\n");
            return out.toOwnedSlice(g);
        }
    };
    // Header says 1 layer of width 2 at rank 1 (44 parameters) but carries one.
    var bad_ad: [28]u8 = @splat(0);
    @memcpy(bad_ad[0..8], "STTLORA1");
    std.mem.writeInt(u32, bad_ad[8..12], 1, .little);
    std.mem.writeInt(u32, bad_ad[12..16], 2, .little);
    std.mem.writeInt(u32, bad_ad[16..20], 1, .little);
    std.mem.writeInt(u32, bad_ad[20..24], @bitCast(@as(f32, 2)), .little);
    const bad = try build.update(gpa, &bad_ad);
    defer gpa.free(bad);
    try std.testing.expectError(error.BadAdapterFile, importUpdate(gpa, &dev, bad));
    try std.testing.expectEqual(@as(u32, 0), dev.version);
    try std.testing.expectEqual(@as(usize, 0), dev.rules.items.len);
    try std.testing.expectError(error.FileNotFound, Io.Dir.cwd().statFile(io, try dev.path("adapter.bin"), .{}));

    // The same header with all 44 parameters is accepted.
    const good_ad = try gpa.alloc(u8, 24 + 44 * 4);
    defer gpa.free(good_ad);
    @memset(good_ad, 0);
    @memcpy(good_ad[0..24], bad_ad[0..24]);
    const good = try build.update(gpa, good_ad);
    defer gpa.free(good);
    try std.testing.expectEqual(@as(u32, 3), try importUpdate(gpa, &dev, good));
    try std.testing.expectEqual(@as(usize, 1), dev.rules.items.len);
    _ = try Io.Dir.cwd().statFile(io, try dev.path("adapter.bin"), .{});
}
