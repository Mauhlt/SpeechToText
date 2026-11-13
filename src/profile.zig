//! Per-user learning profile. Grows from your corrections:
//!
//!   <dir>/vocab.txt    "count<TAB>word"  words you use; fed to the decoder as a prompt
//!   <dir>/rules.txt    "heard => meant"  phrase fixes applied to every transcript
//!   <dir>/samples/N.wav + N.txt        your voice + the right text, for `stt train`
//!   <dir>/samples/N.meta               kind (fix|confirm), heard text, source device
//!   <dir>/adapter.bin                  personal fine-tuned weights (trained on the GPU)
//!   <dir>/state.txt                    learning on/off, adapter version, sync progress
//!
//! vocab.txt and rules.txt are plain text: edit them by hand any time.
const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const audio_mod = @import("audio.zig");

pub const Rule = struct { from: []const u8, to: []const u8 };

/// Samples longer than this aren't kept: training sees one 30 s window.
pub const max_sample_seconds = 30;

pub const SampleKind = enum { confirm, fix };

pub const SampleMeta = struct {
    kind: SampleKind = .confirm,
    /// What the transcriber produced (differs from the text for a fix).
    heard: []const u8 = "",
    /// Device the sample came from ("" = this device).
    source: []const u8 = "",
};

/// A host application's verdict on one transcript.
pub const Feedback = struct {
    /// The correct text (identical to the transcript to confirm it).
    text: []const u8,
    /// false = transcript was wrong and no correct text is known: discard.
    success: bool,
};

pub const LearnResult = struct {
    new_rules: usize = 0,
    new_words: usize = 0,
    sample_id: ?u32 = null,
};

pub const Profile = struct {
    /// Heap-allocated so the arena never moves (the struct is returned by value).
    arena: *std.heap.ArenaAllocator,
    io: Io,
    dir: []const u8,
    words: std.StringArrayHashMapUnmanaged(u32) = .empty,
    rules: std.ArrayList(Rule) = .empty,
    next_sample: u32 = 1,

    // Persisted in state.txt.
    /// false = "good enough": keep and send no data.
    learning: bool = true,
    /// Adapter version in use (devices) / latest trained (home). 0 = base model.
    version: u32 = 0,
    /// Device: samples with id <= exported have been handed to the home device.
    exported: u32 = 0,
    /// Home: samples with id <= trained are already in the adapter.
    trained: u32 = 0,

    /// Opens (creating if needed) the profile directory and loads it.
    pub fn load(gpa: Allocator, io: Io, dir: []const u8) !Profile {
        const arena = try gpa.create(std.heap.ArenaAllocator);
        errdefer gpa.destroy(arena);
        arena.* = .init(gpa);
        var p: Profile = .{ .arena = arena, .io = io, .dir = undefined };
        errdefer p.arena.deinit();
        const a = p.arena.allocator();
        p.dir = try a.dupe(u8, dir);
        const cwd = Io.Dir.cwd();
        try cwd.createDirPath(io, try p.path("samples"));

        if (p.readText("vocab.txt")) |text| try p.mergeVocabText(text);
        if (p.readText("rules.txt")) |text| try p.mergeRulesText(text);
        // Find the next free sample number.
        var sd = try cwd.openDir(io, try p.path("samples"), .{ .iterate = true });
        defer sd.close(io);
        var it = sd.iterate();
        while (try it.next(io)) |e| {
            if (!std.mem.endsWith(u8, e.name, ".txt")) continue;
            const n = std.fmt.parseInt(u32, e.name[0 .. e.name.len - 4], 10) catch continue;
            p.next_sample = @max(p.next_sample, n + 1);
        }
        if (p.readText("state.txt")) |text| {
            var lines = std.mem.tokenizeAny(u8, text, "\r\n");
            while (lines.next()) |line| {
                const eq = std.mem.indexOfScalar(u8, line, '=') orelse continue;
                const k = line[0..eq];
                const v = std.fmt.parseInt(u32, line[eq + 1 ..], 10) catch continue;
                if (std.mem.eql(u8, k, "learning")) p.learning = v != 0;
                if (std.mem.eql(u8, k, "version")) p.version = v;
                if (std.mem.eql(u8, k, "exported")) p.exported = v;
                if (std.mem.eql(u8, k, "trained")) p.trained = v;
            }
        }
        // Exported samples are deleted on devices; never reuse their ids.
        p.next_sample = @max(p.next_sample, @max(p.exported, p.trained) + 1);
        return p;
    }

    pub fn deinit(p: *Profile) void {
        const gpa = p.arena.child_allocator;
        p.arena.deinit();
        gpa.destroy(p.arena);
        p.* = undefined;
    }

    /// Path inside the profile dir. Allocated in the profile's arena.
    pub fn path(p: *Profile, name: []const u8) ![]const u8 {
        return std.fs.path.join(p.arena.allocator(), &.{ p.dir, name });
    }

    fn readText(p: *Profile, name: []const u8) ?[]const u8 {
        const full = p.path(name) catch return null;
        return Io.Dir.cwd().readFileAlloc(p.io, full, p.arena.allocator(), .limited(16 << 20)) catch null;
    }

    /// Prompt text: your explicit prompt plus your most-used learned words.
    pub fn promptText(p: *Profile, gpa: Allocator, base: []const u8) ![]u8 {
        const Ctx = struct {
            vals: []const u32,
            pub fn lessThan(c: @This(), x: usize, y: usize) bool {
                return c.vals[x] > c.vals[y];
            }
        };
        p.words.sort(Ctx{ .vals = p.words.values() });
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        try out.appendSlice(gpa, base);
        for (p.words.keys(), 0..) |w, i| {
            if (i >= 60) break;
            if (out.items.len > 0) try out.appendSlice(gpa, ", ");
            try out.appendSlice(gpa, w);
        }
        return out.toOwnedSlice(gpa);
    }

    /// Applies the learned phrase rules to `text`. Caller owns the result.
    pub fn applyRules(p: *Profile, gpa: Allocator, text: []const u8) ![]u8 {
        if (p.rules.items.len == 0) return gpa.dupe(u8, text);
        var words: std.ArrayList([]const u8) = .empty;
        defer words.deinit(gpa);
        var it = std.mem.tokenizeAny(u8, text, " \t\r\n");
        while (it.next()) |w| try words.append(gpa, w);

        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(gpa);
        var i: usize = 0;
        outer: while (i < words.items.len) {
            if (out.items.len > 0) try out.append(gpa, ' ');
            for (p.rules.items) |r| {
                const n = std.mem.count(u8, r.from, " ") + 1;
                if (i + n > words.items.len) continue;
                if (try phraseMatches(gpa, words.items[i..][0..n], r.from)) {
                    try out.appendSlice(gpa, r.to);
                    // Keep trailing punctuation from the replaced words.
                    const last = words.items[i + n - 1];
                    var k = last.len;
                    while (k > 0 and isTrailPunct(last[k - 1])) k -= 1;
                    try out.appendSlice(gpa, last[k..]);
                    i += n;
                    continue :outer;
                }
            }
            try out.appendSlice(gpa, words.items[i]);
            i += 1;
        }
        return out.toOwnedSlice(gpa);
    }

    /// Merges rules in rules.txt format; incoming rules win on conflicts.
    pub fn mergeRulesText(p: *Profile, text: []const u8) !void {
        const a = p.arena.allocator();
        var lines = std.mem.tokenizeAny(u8, text, "\r\n");
        while (lines.next()) |line| {
            if (line.len == 0 or line[0] == '#') continue;
            const sep = std.mem.indexOf(u8, line, "=>") orelse continue;
            const from_raw = std.mem.trim(u8, line[0..sep], " \t");
            const to = try a.dupe(u8, std.mem.trim(u8, line[sep + 2 ..], " \t"));
            if (from_raw.len == 0) continue;
            const from = try normalizeAlloc(a, from_raw);
            for (p.rules.items) |*r| {
                if (std.mem.eql(u8, r.from, from)) {
                    r.to = to;
                    break;
                }
            } else try p.rules.append(a, .{ .from = from, .to = to });
        }
        p.sortRules();
    }

    /// Merges words in vocab.txt format, keeping the larger count.
    pub fn mergeVocabText(p: *Profile, text: []const u8) !void {
        const a = p.arena.allocator();
        var lines = std.mem.tokenizeAny(u8, text, "\r\n");
        while (lines.next()) |line| {
            const tab = std.mem.indexOfScalar(u8, line, '\t') orelse continue;
            const n = std.fmt.parseInt(u32, line[0..tab], 10) catch continue;
            const w = std.mem.trim(u8, line[tab + 1 ..], " ");
            if (w.len == 0) continue;
            const gop = try p.words.getOrPut(a, w);
            if (!gop.found_existing) {
                gop.key_ptr.* = try a.dupe(u8, w);
                gop.value_ptr.* = n;
            } else gop.value_ptr.* = @max(gop.value_ptr.*, n);
        }
    }

    /// Longer phrases first so they win over their sub-phrases.
    fn sortRules(p: *Profile) void {
        std.mem.sort(Rule, p.rules.items, {}, struct {
            fn lt(_: void, x: Rule, y: Rule) bool {
                return std.mem.count(u8, x.from, " ") > std.mem.count(u8, y.from, " ");
            }
        }.lt);
    }

    pub fn saveState(p: *Profile) !void {
        var buf: [128]u8 = undefined;
        const text = try std.fmt.bufPrint(&buf, "learning={d}\nversion={d}\nexported={d}\ntrained={d}\n", .{
            @intFromBool(p.learning), p.version, p.exported, p.trained,
        });
        try p.writeAtomic("state.txt", text);
    }

    /// Turns learning on or off for this profile ("nothing" response: the
    /// model is good enough here, so keep and send no data).
    pub fn setLearning(p: *Profile, on: bool) !void {
        p.learning = on;
        try p.saveState();
    }

    /// Records host feedback for one utterance. `heard` is the transcript
    /// that was delivered. Returns sample_id = null when no sample was stored.
    pub fn feedback(p: *Profile, gpa: Allocator, audio: []const f32, heard: []const u8, fb: Feedback) !LearnResult {
        return p.feedbackFrom(gpa, audio, heard, fb, "");
    }

    /// Like `feedback`, recording which device the audio came from.
    pub fn feedbackFrom(p: *Profile, gpa: Allocator, audio: []const f32, heard: []const u8, fb: Feedback, source: []const u8) !LearnResult {
        if (!fb.success or !p.learning) return .{};
        const text = std.mem.trim(u8, fb.text, " \t\r\n");
        if (text.len == 0) return .{};
        if (std.mem.eql(u8, text, std.mem.trim(u8, heard, " \t\r\n"))) {
            return .{ .sample_id = try p.saveSampleMeta(gpa, audio, text, .{ .kind = .confirm, .heard = heard, .source = source }) };
        }
        return p.learnFrom(gpa, heard, text, audio, source);
    }

    /// Number of stored training samples.
    pub fn sampleCount(p: *Profile) u32 {
        return p.next_sample - 1;
    }

    /// Learns from a correction: `heard` is what was shown, `meant` what the
    /// user actually said. Saves the audio as a training sample.
    pub fn learn(p: *Profile, gpa: Allocator, heard: []const u8, meant: []const u8, audio: []const f32) !LearnResult {
        return p.learnFrom(gpa, heard, meant, audio, "");
    }

    /// Like `learn`, recording which device the sample came from.
    pub fn learnFrom(p: *Profile, gpa: Allocator, heard: []const u8, meant: []const u8, audio: []const f32, source: []const u8) !LearnResult {
        var res: LearnResult = .{};
        const a = p.arena.allocator();
        var hw = try splitWords(gpa, heard);
        defer hw.deinit(gpa);
        var mw = try splitWords(gpa, meant);
        defer mw.deinit(gpa);

        const gaps = try diffGaps(gpa, hw.items, mw.items);
        defer gpa.free(gaps);
        for (gaps) |g| {
            const h = hw.items[g.h0..g.h1];
            const m = mw.items[g.m0..g.m1];
            for (m) |w| {
                const clean = stripPunct(w.raw);
                if (clean.len < 2) continue;
                const gop = try p.words.getOrPut(a, try a.dupe(u8, clean));
                if (!gop.found_existing) {
                    gop.value_ptr.* = 0;
                    res.new_words += 1;
                }
                gop.value_ptr.* += 1;
            }
            if (h.len == 0 or m.len == 0 or h.len > 4) continue;
            var letters: usize = 0;
            for (h) |w| letters += w.norm.len;
            if (letters < 3) continue;
            const from = try joinNorm(a, h);
            const to = try joinRaw(a, m);
            if (std.ascii.eqlIgnoreCase(from, to)) continue;
            for (p.rules.items) |*r| {
                if (std.mem.eql(u8, r.from, from)) {
                    r.to = to;
                    break;
                }
            } else {
                try p.rules.insert(a, 0, .{ .from = from, .to = to });
                p.sortRules();
                res.new_rules += 1;
            }
        }
        res.sample_id = try p.saveSampleMeta(gpa, audio, meant, .{ .kind = .fix, .heard = heard, .source = source });
        try p.save(gpa);
        return res;
    }

    /// Saves (audio, text) as a confirmed-correct training sample.
    pub fn saveSample(p: *Profile, gpa: Allocator, audio: []const f32, text: []const u8) !u32 {
        return (try p.saveSampleMeta(gpa, audio, text, .{ .heard = text })) orelse error.SampleTooLong;
    }

    /// Saves a sample with metadata. Returns null (nothing saved) when the
    /// audio is longer than `max_sample_seconds`.
    pub fn saveSampleMeta(p: *Profile, gpa: Allocator, audio: []const f32, text: []const u8, meta: SampleMeta) !?u32 {
        if (audio.len > max_sample_seconds * audio_mod.target_rate) return null;
        const id = p.next_sample;
        p.next_sample += 1;
        var name_buf: [32]u8 = undefined;
        const bytes = try audio_mod.encodeWav(gpa, audio, audio_mod.target_rate);
        defer gpa.free(bytes);
        const cwd = Io.Dir.cwd();
        try cwd.writeFile(p.io, .{ .sub_path = try p.path(try std.fmt.bufPrint(&name_buf, "samples/{d:0>6}.wav", .{id})), .data = bytes });
        const meta_text = try std.fmt.allocPrint(gpa, "kind={t}\nsource={s}\nheard={s}\n", .{ meta.kind, oneLine(meta.source), oneLine(meta.heard) });
        defer gpa.free(meta_text);
        try cwd.writeFile(p.io, .{ .sub_path = try p.path(try std.fmt.bufPrint(&name_buf, "samples/{d:0>6}.meta", .{id})), .data = meta_text });
        // The .txt is written last: its presence marks a complete sample.
        try p.writeAtomic(try std.fmt.bufPrint(&name_buf, "samples/{d:0>6}.txt", .{id}), text);
        return id;
    }

    pub fn save(p: *Profile, gpa: Allocator) !void {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(gpa);
        for (p.words.keys(), p.words.values()) |w, n| {
            try buf.print(gpa, "{d}\t{s}\n", .{ n, w });
        }
        try p.writeAtomic("vocab.txt", buf.items);
        buf.clearRetainingCapacity();
        try buf.appendSlice(gpa, "# heard => meant   (learned from your corrections; edit freely)\n");
        for (p.rules.items) |r| try buf.print(gpa, "{s} => {s}\n", .{ r.from, r.to });
        try p.writeAtomic("rules.txt", buf.items);
    }

    /// Writes `name` via a temp file + rename, so a crash mid-write leaves
    /// the old contents instead of a truncated file.
    pub fn writeAtomic(p: *Profile, name: []const u8, data: []const u8) !void {
        var buf: [96]u8 = undefined;
        const tmp_name = try std.fmt.bufPrint(&buf, "{s}.tmp", .{name});
        const cwd = Io.Dir.cwd();
        const tmp = try p.path(tmp_name);
        try cwd.writeFile(p.io, .{ .sub_path = tmp, .data = data });
        try cwd.rename(tmp, cwd, try p.path(name), p.io);
    }
};

/// Metadata values must stay on one line.
fn oneLine(s: []const u8) []const u8 {
    const end = std.mem.indexOfAny(u8, s, "\r\n") orelse s.len;
    return s[0..end];
}

const Word = struct { raw: []const u8, norm: []const u8 };

const WordList = struct {
    items: []Word,
    norm_buf: []u8,
    fn deinit(w: *WordList, gpa: Allocator) void {
        gpa.free(w.items);
        gpa.free(w.norm_buf);
    }
};

fn splitWords(gpa: Allocator, text: []const u8) !WordList {
    var list: std.ArrayList(Word) = .empty;
    errdefer list.deinit(gpa);
    const nb = try gpa.alloc(u8, text.len);
    errdefer gpa.free(nb);
    var used: usize = 0;
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n");
    while (it.next()) |raw| {
        const start = used;
        for (raw) |c| if (std.ascii.isAlphanumeric(c) or c == '\'' or c >= 0x80) {
            nb[used] = std.ascii.toLower(c);
            used += 1;
        };
        try list.append(gpa, .{ .raw = raw, .norm = nb[start..used] });
    }
    return .{ .items = try list.toOwnedSlice(gpa), .norm_buf = nb };
}

const Gap = struct { h0: usize, h1: usize, m0: usize, m1: usize };

/// Longest-common-subsequence alignment of two word lists; returns the
/// unmatched stretches between matches.
fn diffGaps(gpa: Allocator, h: []const Word, m: []const Word) ![]Gap {
    const W = m.len + 1;
    const dp = try gpa.alloc(u32, (h.len + 1) * W);
    defer gpa.free(dp);
    @memset(dp, 0);
    var i = h.len;
    while (i > 0) {
        i -= 1;
        var j = m.len;
        while (j > 0) {
            j -= 1;
            dp[i * W + j] = if (std.mem.eql(u8, h[i].norm, m[j].norm))
                dp[(i + 1) * W + j + 1] + 1
            else
                @max(dp[(i + 1) * W + j], dp[i * W + j + 1]);
        }
    }
    var gaps: std.ArrayList(Gap) = .empty;
    errdefer gaps.deinit(gpa);
    var a: usize = 0;
    var b: usize = 0;
    var g: Gap = .{ .h0 = 0, .h1 = 0, .m0 = 0, .m1 = 0 };
    while (a < h.len or b < m.len) {
        if (a < h.len and b < m.len and std.mem.eql(u8, h[a].norm, m[b].norm)) {
            if (g.h1 > g.h0 or g.m1 > g.m0) try gaps.append(gpa, g);
            a += 1;
            b += 1;
            g = .{ .h0 = a, .h1 = a, .m0 = b, .m1 = b };
        } else if (b >= m.len or (a < h.len and dp[(a + 1) * W + b] >= dp[a * W + b + 1])) {
            a += 1;
            g.h1 = a;
        } else {
            b += 1;
            g.m1 = b;
        }
    }
    if (g.h1 > g.h0 or g.m1 > g.m0) try gaps.append(gpa, g);
    return gaps.toOwnedSlice(gpa);
}

fn isTrailPunct(c: u8) bool {
    return c == '.' or c == ',' or c == '!' or c == '?' or c == ';' or c == ':';
}

fn stripPunct(w: []const u8) []const u8 {
    return std.mem.trim(u8, w, ".,!?;:\"()[]");
}

fn joinNorm(a: Allocator, ws: []const Word) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (ws, 0..) |w, i| {
        if (i > 0) try out.append(a, ' ');
        try out.appendSlice(a, w.norm);
    }
    return out.toOwnedSlice(a);
}

fn joinRaw(a: Allocator, ws: []const Word) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (ws, 0..) |w, i| {
        if (i > 0) try out.append(a, ' ');
        const s = if (i == ws.len - 1) std.mem.trimEnd(u8, w.raw, ".,!?;:") else w.raw;
        try out.appendSlice(a, s);
    }
    return out.toOwnedSlice(a);
}

fn normalizeAlloc(a: Allocator, phrase: []const u8) ![]const u8 {
    const wl = try splitWords(a, phrase);
    return joinNorm(a, wl.items);
}

fn phraseMatches(gpa: Allocator, words: []const []const u8, from: []const u8) !bool {
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(gpa);
    for (words, 0..) |w, i| {
        if (i > 0) try buf.append(gpa, ' ');
        for (w) |c| if (std.ascii.isAlphanumeric(c) or c == '\'' or c >= 0x80) try buf.append(gpa, std.ascii.toLower(c));
    }
    return std.mem.eql(u8, buf.items, from);
}

test "learn rule from correction and apply it" {
    const gpa = std.testing.allocator;
    var h = try splitWords(gpa, "written in Puresig.");
    defer h.deinit(gpa);
    var m = try splitWords(gpa, "written in pure Zig.");
    defer m.deinit(gpa);
    const gaps = try diffGaps(gpa, h.items, m.items);
    defer gpa.free(gaps);
    try std.testing.expectEqual(@as(usize, 1), gaps.len);
    try std.testing.expectEqual(@as(usize, 2), gaps[0].h0);
    try std.testing.expectEqual(@as(usize, 3), gaps[0].h1);
    try std.testing.expectEqual(@as(usize, 2), gaps[0].m0);
    try std.testing.expectEqual(@as(usize, 4), gaps[0].m1);
}

test "feedback: confirm, correct, discard" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);

    var p = try Profile.load(gpa, io, dir);
    defer p.deinit();
    const audio: [1600]f32 = @splat(0);

    // Wrong, no correct text: nothing stored.
    const r0 = try p.feedback(gpa, &audio, "Learning's egg.", .{ .text = "", .success = false });
    try std.testing.expect(r0.sample_id == null);
    // Confirmed as-is: stored, no rules.
    const r1 = try p.feedback(gpa, &audio, "Hello world.", .{ .text = "Hello world.", .success = true });
    try std.testing.expectEqual(@as(?u32, 1), r1.sample_id);
    try std.testing.expectEqual(@as(usize, 0), r1.new_rules);
    // Corrected: stored + rule learned + applied next time.
    const r2 = try p.feedback(gpa, &audio, "written in Puresig.", .{ .text = "written in pure Zig.", .success = true });
    try std.testing.expectEqual(@as(?u32, 2), r2.sample_id);
    try std.testing.expectEqual(@as(usize, 1), r2.new_rules);
    const fixed = try p.applyRules(gpa, "It is written in Puresig, really.");
    defer gpa.free(fixed);
    try std.testing.expectEqualStrings("It is written in pure Zig, really.", fixed);

    // Reloading from disk keeps what was learned.
    var p2 = try Profile.load(gpa, io, dir);
    defer p2.deinit();
    try std.testing.expectEqual(@as(usize, 1), p2.rules.items.len);
    try std.testing.expectEqual(@as(u32, 3), p2.next_sample);
}
