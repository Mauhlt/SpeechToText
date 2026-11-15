//! Per-user learning profile. Grows from your corrections:
//!
//!   <dir>/vocab.txt    "count<TAB>word"  words you use; fed to the decoder as a prompt
//!   <dir>/rules.txt    "heard => meant"  phrase fixes applied to every transcript
//!   <dir>/sources.txt  "w|r<TAB>word or heard<TAB>device"  which device taught a
//!                      word or rule (absent = this device or edited by hand)
//!   <dir>/samples/N.wav + N.txt        your voice + the right text, for `stt train`
//!   <dir>/samples/N.meta               kind (fix|confirm), heard text, source device
//!   <dir>/adapter.bin                  personal fine-tuned weights (trained on the GPU)
//!   <dir>/state.txt                    learning on/off, adapter version, sync progress
//!
//! vocab.txt and rules.txt are plain text: edit them by hand any time.
//!
//! A sample is complete once its .txt exists (written last, deleted first);
//! a .wav or .meta without one is a leftover that loading, training and
//! export ignore and `deleteOldestSamples` removes.
const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const audio_mod = @import("audio.zig");

pub const Rule = struct {
    from: []const u8,
    to: []const u8,
    /// Device that taught it ("" = this device or edited by hand).
    source: []const u8 = "",
};

/// Samples longer than this aren't kept: training sees one 30 s window.
pub const max_sample_seconds = 30;

/// Caps on what one correction can teach. The alignment is quadratic in
/// the word count, so longer texts are stored as samples but teach nothing.
pub const max_learn_words = 256;
/// Longer words are never learned as vocabulary or used in a rule.
pub const max_word_bytes = 64;
/// Longest replacement text a learned rule may have.
pub const max_rule_bytes = 256;
/// Most rules and words one correction can propose.
pub const max_lesson_rules = 16;
pub const max_lesson_words = 64;
/// Caps on the learned set as a whole: learning stops adding (it still
/// updates what exists) once they are reached. Hand-edited files may hold more.
pub const max_rules = 2048;
pub const max_words = 20_000;
/// Longest device name recorded as a source.
pub const max_source_bytes = 128;

pub const SampleKind = enum { confirm, fix };

pub const SampleMeta = struct {
    kind: SampleKind = .confirm,
    /// What the transcriber produced (differs from the text for a fix).
    heard: []const u8 = "",
    /// Device the sample came from ("" = this device).
    source: []const u8 = "",

    /// Parses a sample's .meta file. The result points into `text`.
    pub fn parse(text: []const u8) SampleMeta {
        var m: SampleMeta = .{};
        var lines = std.mem.tokenizeAny(u8, text, "\r\n");
        while (lines.next()) |line| {
            const eq = std.mem.indexOfScalar(u8, line, '=') orelse continue;
            const k = line[0..eq];
            const v = line[eq + 1 ..];
            if (std.mem.eql(u8, k, "kind")) m.kind = std.meta.stringToEnum(SampleKind, v) orelse .confirm;
            if (std.mem.eql(u8, k, "heard")) m.heard = v;
            if (std.mem.eql(u8, k, "source")) m.source = v;
        }
        return m;
    }
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

/// What one correction teaches, worked out without touching a profile (so
/// it can be computed outside any lock). See `lesson`.
pub const Lesson = struct {
    /// Candidate phrase rules: `from` normalized, `to` as written.
    rules: []const Rule = &.{},
    /// Candidate vocabulary words, one per use.
    words: []const []const u8 = &.{},
};

/// How many new rules and words a batch of lessons may still add.
pub const Budget = struct {
    rules: usize = std.math.maxInt(usize),
    words: usize = std.math.maxInt(usize),
};

/// Storage report for `deleteOldestSamples` (and `sync.compact`).
pub const StorageReport = struct {
    freed_bytes: u64 = 0,
    deleted_samples: u32 = 0,
    bytes_after: u64 = 0,
};

pub const ForgetReport = struct {
    samples: u32 = 0,
    rules: u32 = 0,
    words: u32 = 0,
};

pub const Profile = struct {
    /// Heap-allocated so the arena never moves (the struct is returned by value).
    arena: *std.heap.ArenaAllocator,
    io: Io,
    dir: []const u8,
    words: std.StringArrayHashMapUnmanaged(u32) = .empty,
    /// The device that taught each learned word (keys are `words`' keys;
    /// absent = this device or edited by hand).
    word_sources: std.StringHashMapUnmanaged([]const u8) = .empty,
    rules: std.ArrayList(Rule) = .empty,
    /// Interned source names (sources are shared by many words and rules).
    source_names: std.StringHashMapUnmanaged(void) = .empty,
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
        if (p.readText("sources.txt")) |text| try p.loadSources(text);
        // Find the next free sample number (leftover files of an
        // incomplete sample count too, so their id is never reused).
        var sd = try cwd.openDir(io, try p.path("samples"), .{ .iterate = true });
        defer sd.close(io);
        var it = sd.iterate();
        while (try it.next(io)) |e| {
            const n = sampleIdOf(e.name) orelse continue;
            p.next_sample = @max(p.next_sample, n +| 1);
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
        p.next_sample = @max(p.next_sample, @max(p.exported, p.trained) +| 1);
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
            if (p.findRule(from)) |r| {
                r.to = to;
                r.source = "";
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

    /// Attaches sources.txt's provenance to the loaded words and rules.
    fn loadSources(p: *Profile, text: []const u8) !void {
        var lines = std.mem.tokenizeAny(u8, text, "\r\n");
        while (lines.next()) |line| {
            var f = std.mem.splitScalar(u8, line, '\t');
            const kind = f.next() orelse continue;
            const key = f.next() orelse continue;
            const src = try p.internSource(f.next() orelse continue);
            if (src.len == 0) continue;
            if (std.mem.eql(u8, kind, "w")) {
                if (p.words.getKey(key)) |k| try p.word_sources.put(p.arena.allocator(), k, src);
            } else if (std.mem.eql(u8, kind, "r")) {
                if (p.findRule(key)) |r| r.source = src;
            }
        }
    }

    fn findRule(p: *Profile, from: []const u8) ?*Rule {
        for (p.rules.items) |*r| if (std.mem.eql(u8, r.from, from)) return r;
        return null;
    }

    /// The profile's copy of a (cleaned) source name; "" stays "".
    fn internSource(p: *Profile, source: []const u8) ![]const u8 {
        var buf: [max_source_bytes]u8 = undefined;
        const clean = cleanInto(&buf, source);
        if (clean.len == 0) return "";
        const a = p.arena.allocator();
        const gop = try p.source_names.getOrPut(a, clean);
        if (!gop.found_existing) gop.key_ptr.* = try a.dupe(u8, clean);
        return gop.key_ptr.*;
    }

    /// Longer phrases first so they win over their sub-phrases.
    pub fn sortRules(p: *Profile) void {
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
    /// Control characters in the texts become spaces.
    pub fn feedbackFrom(p: *Profile, gpa: Allocator, audio: []const f32, heard_raw: []const u8, fb: Feedback, source: []const u8) !LearnResult {
        if (!fb.success or !p.learning) return .{};
        const text = try cleanText(gpa, fb.text);
        defer gpa.free(text);
        if (text.len == 0) return .{};
        const heard = try cleanText(gpa, heard_raw);
        defer gpa.free(heard);
        if (std.mem.eql(u8, text, heard)) {
            return .{ .sample_id = try p.saveSampleMeta(gpa, audio, text, .{ .kind = .confirm, .heard = heard, .source = source }) };
        }
        return p.learnFrom(gpa, heard, text, audio, source);
    }

    /// Learns from a correction: `heard` is what was shown, `meant` what the
    /// user actually said. Saves the audio as a training sample.
    fn learnFrom(p: *Profile, gpa: Allocator, heard: []const u8, meant: []const u8, audio: []const f32, source: []const u8) !LearnResult {
        var tmp: std.heap.ArenaAllocator = .init(gpa);
        defer tmp.deinit();
        const les = try lesson(tmp.allocator(), heard, meant);
        var budget: Budget = .{};
        var res = try p.applyLesson(les, source, &budget);
        p.sortRules();
        res.sample_id = try p.saveSampleMeta(gpa, audio, meant, .{ .kind = .fix, .heard = heard, .source = source });
        try p.save(gpa);
        return res;
    }

    /// Adds a lesson's words and rules, recording `source` for new ones and
    /// for rules whose replacement it changes. Existing words are only
    /// counted (nothing is copied); new words, and new or changed rules,
    /// stop at `budget` and at `max_words` / `max_rules`. Leaves the rules unsorted and nothing
    /// written: call `sortRules` and `save` once after a batch.
    pub fn applyLesson(p: *Profile, les: Lesson, source: []const u8, budget: *Budget) !LearnResult {
        const a = p.arena.allocator();
        const src = try p.internSource(source);
        var res: LearnResult = .{};
        for (les.words) |w| {
            if (p.words.getPtr(w)) |n| {
                n.* +|= 1;
                continue;
            }
            if (p.words.count() >= max_words or budget.words == 0) continue;
            const key = try a.dupe(u8, w);
            try p.words.put(a, key, 1);
            if (src.len > 0) try p.word_sources.put(a, key, src);
            budget.words -= 1;
            res.new_words += 1;
        }
        for (les.rules) |r| {
            if (p.findRule(r.from)) |old| {
                // Changing a rule uses the budget too, so one bundle can't
                // rewrite every rule.
                if (!std.mem.eql(u8, old.to, r.to) and budget.rules > 0) {
                    old.to = try a.dupe(u8, r.to);
                    old.source = src;
                    budget.rules -= 1;
                }
                continue;
            }
            if (p.rules.items.len >= max_rules or budget.rules == 0) continue;
            try p.rules.append(a, .{ .from = try a.dupe(u8, r.from), .to = try a.dupe(u8, r.to), .source = src });
            budget.rules -= 1;
            res.new_rules += 1;
        }
        return res;
    }

    /// Saves a sample with metadata. Returns null (nothing saved) when the
    /// audio is longer than `max_sample_seconds`.
    pub fn saveSampleMeta(p: *Profile, gpa: Allocator, audio: []const f32, text: []const u8, meta: SampleMeta) !?u32 {
        if (audio.len > max_sample_seconds * audio_mod.target_rate) return null;
        const bytes = try audio_mod.encodeWav(gpa, audio, audio_mod.target_rate);
        defer gpa.free(bytes);
        return try p.writeSample(bytes, text, meta);
    }

    /// Stores an encoded sample (16 kHz mono WAV) under the next id.
    pub fn writeSample(p: *Profile, wav: []const u8, text: []const u8, meta: SampleMeta) !u32 {
        if (p.next_sample == std.math.maxInt(u32)) return error.ProfileFull;
        const id = p.next_sample;
        p.next_sample += 1;
        var name_buf: [32]u8 = undefined;
        var meta_buf: [max_source_bytes]u8 = undefined;
        const cwd = Io.Dir.cwd();
        try cwd.writeFile(p.io, .{ .sub_path = try p.path(try std.fmt.bufPrint(&name_buf, "samples/{d:0>6}.wav", .{id})), .data = wav });
        const gpa = p.arena.child_allocator;
        const meta_text = try std.fmt.allocPrint(gpa, "kind={t}\nsource={s}\nheard={s}\n", .{ meta.kind, cleanInto(&meta_buf, meta.source), oneLine(meta.heard) });
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
        buf.clearRetainingCapacity();
        for (p.words.keys()) |w| if (p.word_sources.get(w)) |s| try buf.print(gpa, "w\t{s}\t{s}\n", .{ w, s });
        for (p.rules.items) |r| if (r.source.len > 0) try buf.print(gpa, "r\t{s}\t{s}\n", .{ r.from, r.source });
        try p.writeAtomic("sources.txt", buf.items);
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

    // ------------------------------------------------------------ storage

    /// Bytes of every file in the profile folder: samples, vocabulary,
    /// rules, state, the adapter and its archive (and, on a laptop, the
    /// NPU's compiled-graph cache when it shares the folder).
    pub fn profileBytes(p: *Profile) !u64 {
        const gpa = p.arena.child_allocator;
        var dir = try Io.Dir.cwd().openDir(p.io, p.dir, .{ .iterate = true });
        defer dir.close(p.io);
        var w = try dir.walk(gpa);
        defer w.deinit();
        var total: u64 = 0;
        while (try w.next(p.io)) |e| {
            if (e.kind != .file) continue;
            const st = e.dir.statFile(p.io, e.basename, .{}) catch continue;
            total += st.size;
        }
        return total;
    }

    /// Storage cap: deletes whole samples (audio, text and metadata), oldest
    /// id first, until `profileBytes()` is at most `target_bytes` or no
    /// samples are left. The vocabulary, rules, state, adapter and its
    /// archive are never touched, so the result can stay above the target
    /// (`bytes_after` says).
    ///
    /// Interruption-safe: each sample loses its .txt first (which marks it
    /// incomplete, so loading, training and export skip it), then its .wav
    /// and .meta; a later call removes any leftovers along with the rest
    /// (they count as samples, in id order).
    ///
    /// Safe while a training run is going: training reads each sample once
    /// when it starts and skips any that are gone (see `training.loadSamples`).
    /// Untrained samples deleted here are simply never trained on.
    pub fn deleteOldestSamples(p: *Profile, target_bytes: u64) !StorageReport {
        var rep: StorageReport = .{};
        var size = try p.profileBytes();
        if (size > target_bytes) {
            const gpa = p.arena.child_allocator;
            const ids = try p.sampleIds(gpa);
            defer gpa.free(ids);
            for (ids) |id| {
                if (size <= target_bytes) break;
                const freed = p.deleteSample(id) catch continue;
                size -|= freed;
                rep.freed_bytes += freed;
                rep.deleted_samples += 1;
            }
        }
        rep.bytes_after = size;
        return rep;
    }

    /// Forgets everything device `name` taught: its samples (all files),
    /// and the rules and words it added. What was already trained into the
    /// adapter stays there until the next training run without it.
    pub fn forgetSource(p: *Profile, name: []const u8) !ForgetReport {
        var buf: [max_source_bytes]u8 = undefined;
        const want = cleanInto(&buf, name);
        if (want.len == 0) return error.InvalidSource;
        const gpa = p.arena.child_allocator;
        var rep: ForgetReport = .{};

        const ids = try p.sampleIds(gpa);
        defer gpa.free(ids);
        var name_buf: [32]u8 = undefined;
        for (ids) |id| {
            const rel = std.fmt.bufPrint(&name_buf, "samples/{d:0>6}.meta", .{id}) catch unreachable;
            const meta_text = Io.Dir.cwd().readFileAlloc(p.io, try p.path(rel), gpa, .limited(64 * 1024)) catch continue;
            defer gpa.free(meta_text);
            if (!std.mem.eql(u8, SampleMeta.parse(meta_text).source, want)) continue;
            _ = try p.deleteSample(id);
            rep.samples += 1;
        }

        var i: usize = 0;
        while (i < p.rules.items.len) {
            if (std.mem.eql(u8, p.rules.items[i].source, want)) {
                _ = p.rules.orderedRemove(i); // keeps the longest-first order
                rep.rules += 1;
            } else i += 1;
        }

        var drop: std.ArrayList([]const u8) = .empty;
        defer drop.deinit(gpa);
        var it = p.word_sources.iterator();
        while (it.next()) |e| if (std.mem.eql(u8, e.value_ptr.*, want)) try drop.append(gpa, e.key_ptr.*);
        for (drop.items) |w| {
            _ = p.word_sources.remove(w);
            _ = p.words.swapRemove(w);
            rep.words += 1;
        }
        try p.save(gpa);
        return rep;
    }

    /// Deletes one sample's files, .txt first. Returns the bytes freed.
    fn deleteSample(p: *Profile, id: u32) !u64 {
        var freed: u64 = 0;
        var name_buf: [48]u8 = undefined;
        const cwd = Io.Dir.cwd();
        for ([_][]const u8{ "txt", "wav", "meta", "txt.tmp" }) |ext| {
            const rel = std.fmt.bufPrint(&name_buf, "samples/{d:0>6}.{s}", .{ id, ext }) catch unreachable;
            const full = try p.path(rel);
            const st = cwd.statFile(p.io, full, .{}) catch |err| switch (err) {
                error.FileNotFound => continue,
                else => return err,
            };
            cwd.deleteFile(p.io, full) catch |err| switch (err) {
                error.FileNotFound => continue,
                else => return err,
            };
            freed += st.size;
        }
        return freed;
    }

    /// Ids of every sample with any file on disk, ascending. Caller frees.
    fn sampleIds(p: *Profile, gpa: Allocator) ![]u32 {
        var dir = try Io.Dir.cwd().openDir(p.io, try p.path("samples"), .{ .iterate = true });
        defer dir.close(p.io);
        var list: std.ArrayList(u32) = .empty;
        errdefer list.deinit(gpa);
        var it = dir.iterate();
        while (try it.next(p.io)) |e| {
            if (e.kind != .file) continue;
            try list.append(gpa, sampleIdOf(e.name) orelse continue);
        }
        std.mem.sort(u32, list.items, {}, std.sort.asc(u32));
        // Each id appears once per file; keep one.
        var n: usize = 0;
        for (list.items) |id| {
            if (n > 0 and list.items[n - 1] == id) continue;
            list.items[n] = id;
            n += 1;
        }
        list.shrinkRetainingCapacity(n);
        return list.toOwnedSlice(gpa);
    }
};

/// The sample id of a file in samples/ ("000012.wav" -> 12), or null.
fn sampleIdOf(name: []const u8) ?u32 {
    const base = if (std.mem.endsWith(u8, name, ".tmp")) name[0 .. name.len - 4] else name;
    const dot = std.mem.lastIndexOfScalar(u8, base, '.') orelse return null;
    const ext = base[dot + 1 ..];
    if (!std.mem.eql(u8, ext, "txt") and !std.mem.eql(u8, ext, "wav") and !std.mem.eql(u8, ext, "meta")) return null;
    return std.fmt.parseInt(u32, base[0..dot], 10) catch null;
}

fn isControl(c: u8) bool {
    return c < 0x20 or c == 0x7f;
}

/// Text from a host or another device, made safe to learn from and store:
/// control characters (line breaks, tabs, escapes) become spaces, and the
/// ends are trimmed. Caller owns the result.
pub fn cleanText(a: Allocator, s: []const u8) ![]u8 {
    var start: usize = 0;
    var end = s.len;
    while (start < end and (s[start] == ' ' or isControl(s[start]))) start += 1;
    while (end > start and (s[end - 1] == ' ' or isControl(s[end - 1]))) end -= 1;
    const out = try a.dupe(u8, s[start..end]);
    for (out) |*c| if (isControl(c.*)) {
        c.* = ' ';
    };
    return out;
}

/// `cleanText` into a fixed buffer (cut to its size), for source names.
fn cleanInto(buf: []u8, s: []const u8) []const u8 {
    var n: usize = 0;
    for (s) |c| {
        if (n == buf.len) break;
        buf[n] = if (isControl(c)) ' ' else c;
        n += 1;
    }
    return std.mem.trim(u8, buf[0..n], " ");
}

/// A device name as recorded for provenance: cleaned, 1..max_source_bytes.
pub fn checkSource(name: []const u8) !void {
    var buf: [max_source_bytes]u8 = undefined;
    if (name.len > max_source_bytes or cleanInto(&buf, name).len == 0) return error.InvalidSource;
}

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

/// Works out what correcting `heard` to `meant` teaches: the words of
/// `meant` in each changed stretch, and a phrase rule for each short
/// changed stretch. Pure (no profile), bounded: texts over
/// `max_learn_words` teach nothing, words over `max_word_bytes` are never
/// learned. Allocates in `a` (use an arena); the result may point into
/// `meant`.
pub fn lesson(a: Allocator, heard: []const u8, meant: []const u8) !Lesson {
    const hw = try splitWords(a, heard);
    const mw = try splitWords(a, meant);
    if (hw.items.len > max_learn_words or mw.items.len > max_learn_words) return .{};
    var rules: std.ArrayList(Rule) = .empty;
    var words: std.ArrayList([]const u8) = .empty;
    const gaps = try diffGaps(a, hw.items, mw.items);
    for (gaps) |g| {
        const h = hw.items[g.h0..g.h1];
        const m = mw.items[g.m0..g.m1];
        for (m) |w| {
            const clean = stripPunct(w.raw);
            if (clean.len < 2 or clean.len > max_word_bytes) continue;
            if (words.items.len < max_lesson_words) try words.append(a, clean);
        }
        if (h.len == 0 or m.len == 0 or h.len > 4) continue;
        if (rules.items.len >= max_lesson_rules) continue;
        var letters: usize = 0;
        var too_long = false;
        for (h) |w| {
            letters += w.norm.len;
            too_long = too_long or w.raw.len > max_word_bytes;
        }
        for (m) |w| too_long = too_long or w.raw.len > max_word_bytes;
        if (letters < 3 or too_long) continue;
        const from = try joinNorm(a, h);
        const to = try joinRaw(a, m);
        if (to.len > max_rule_bytes or std.ascii.eqlIgnoreCase(from, to)) continue;
        try rules.append(a, .{ .from = from, .to = to });
    }
    return .{ .rules = rules.items, .words = words.items };
}

const Gap = struct { h0: usize, h1: usize, m0: usize, m1: usize };

/// Longest-common-subsequence alignment of two word lists; returns the
/// unmatched stretches between matches. Quadratic: callers bound the lists
/// (`max_learn_words`).
fn diffGaps(gpa: Allocator, h: []const Word, m: []const Word) ![]Gap {
    std.debug.assert(h.len <= max_learn_words and m.len <= max_learn_words);
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

// ------------------------------------------------------------------ tests

/// Test helper: stores a confirmed sample (the old `Profile.saveSample`).
pub fn testSample(p: *Profile, gpa: Allocator, audio: []const f32, text: []const u8) !u32 {
    if (!@import("builtin").is_test) @compileError("test helper");
    return (try p.saveSampleMeta(gpa, audio, text, .{ .heard = text })) orelse error.SampleTooLong;
}

fn testDir(gpa: Allocator, tmp: *std.testing.TmpDir) ![]u8 {
    return std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
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
    const dir = try testDir(gpa, &tmp);
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

test "lesson: bounded, no huge words, control characters cleaned" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Over max_learn_words on either side: teaches nothing (and never
    // allocates the quadratic alignment table).
    var long: std.ArrayList(u8) = .empty;
    for (0..max_learn_words + 1) |i| try long.print(a, "w{d} ", .{i});
    const none = try lesson(a, long.items, "something else entirely");
    try std.testing.expectEqual(@as(usize, 0), none.rules.len + none.words.len);
    const none2 = try lesson(a, "short", long.items);
    try std.testing.expectEqual(@as(usize, 0), none2.rules.len + none2.words.len);

    // A word over max_word_bytes is neither vocabulary nor part of a rule.
    const big: [max_word_bytes + 1]u8 = @splat('x');
    const l = try lesson(a, "call Bobby now", "call " ++ &big ++ " now");
    try std.testing.expectEqual(@as(usize, 0), l.rules.len);
    try std.testing.expectEqual(@as(usize, 0), l.words.len);

    const clean = try cleanText(a, "\x1b[31mred\x00 text\r\n");
    try std.testing.expectEqualStrings("[31mred  text", clean);
}

test "provenance: sources are recorded, persisted, and forgotten" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try testDir(gpa, &tmp);
    defer gpa.free(dir);
    const audio: [1600]f32 = @splat(0);
    {
        var p = try Profile.load(gpa, io, dir);
        defer p.deinit();
        _ = try p.feedbackFrom(gpa, &audio, "written in Puresig.", .{ .text = "written in pure Zig.", .success = true }, "laptop");
        _ = try p.feedbackFrom(gpa, &audio, "meet Bobbie today", .{ .text = "meet Bobby today", .success = true }, "phone");
        _ = try p.feedback(gpa, &audio, "my own sample", .{ .text = "my own sample", .success = true });
        try std.testing.expectEqual(@as(usize, 2), p.rules.items.len);
    }
    // Reload: provenance survived; forget the laptop.
    var p = try Profile.load(gpa, io, dir);
    defer p.deinit();
    try std.testing.expectEqualStrings("laptop", p.findRule("puresig").?.source);
    try std.testing.expectEqualStrings("phone", p.word_sources.get("Bobby").?);
    try std.testing.expectError(error.InvalidSource, p.forgetSource(" \t"));
    const rep = try p.forgetSource("laptop");
    try std.testing.expectEqual(@as(u32, 1), rep.samples);
    try std.testing.expectEqual(@as(u32, 1), rep.rules);
    try std.testing.expectEqual(@as(u32, 2), rep.words); // "pure", "Zig"
    try std.testing.expect(p.findRule("puresig") == null);
    try std.testing.expect(p.findRule("bobbie") != null);
    try std.testing.expect(p.words.get("Zig") == null);
    try std.testing.expect(p.words.get("Bobby") != null);
    // The laptop's sample (id 1) is gone; the others stay.
    var nb: [32]u8 = undefined;
    try std.testing.expectError(error.FileNotFound, Io.Dir.cwd().statFile(io, try p.path(try std.fmt.bufPrint(&nb, "samples/{d:0>6}.txt", .{1})), .{}));
    _ = try Io.Dir.cwd().statFile(io, try p.path(try std.fmt.bufPrint(&nb, "samples/{d:0>6}.txt", .{2})), .{});
    _ = try Io.Dir.cwd().statFile(io, try p.path(try std.fmt.bufPrint(&nb, "samples/{d:0>6}.txt", .{3})), .{});
    var p2 = try Profile.load(gpa, io, dir);
    defer p2.deinit();
    try std.testing.expectEqual(@as(usize, 1), p2.rules.items.len);
}

test "learning caps: totals stop growth, existing words are counted not copied" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try testDir(gpa, &tmp);
    defer gpa.free(dir);
    var p = try Profile.load(gpa, io, dir);
    defer p.deinit();

    var budget: Budget = .{ .rules = 1, .words = 1 };
    const les: Lesson = .{
        .rules = &.{ .{ .from = "aaa", .to = "A" }, .{ .from = "bbb", .to = "B" } },
        .words = &.{ "one", "two", "one" },
    };
    const r = try p.applyLesson(les, "dev", &budget);
    try std.testing.expectEqual(@as(usize, 1), r.new_rules);
    try std.testing.expectEqual(@as(usize, 1), r.new_words);
    try std.testing.expectEqual(@as(u32, 2), p.words.get("one").?);
    // Re-learning existing words and rules copies nothing (1000 copies of
    // a 60-byte word would grow the arena by 60 KB).
    const long_arr: [60]u8 = @splat('w');
    const long: []const u8 = &long_arr;
    var more: Budget = .{};
    _ = try p.applyLesson(.{ .words = &.{long} }, "dev", &more);
    const used = p.arena.queryCapacity();
    for (0..1000) |_| _ = try p.applyLesson(.{ .words = &.{ long, "one" }, .rules = &.{.{ .from = "aaa", .to = "A" }} }, "dev", &more);
    try std.testing.expectEqual(used, p.arena.queryCapacity());
    try std.testing.expectEqual(@as(u32, 1002), p.words.get("one").?);
}

test "deleteOldestSamples: shrinks to the target, keeps the newest, survives interruption" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try testDir(gpa, &tmp);
    defer gpa.free(dir);
    const audio: [16000]f32 = @splat(0.01); // ~32 KB per sample
    var nb: [32]u8 = undefined;
    {
        var p = try Profile.load(gpa, io, dir);
        defer p.deinit();
        _ = try p.feedback(gpa, &audio, "written in Puresig.", .{ .text = "written in pure Zig.", .success = true });
        for (0..9) |_| _ = try testSample(&p, gpa, &audio, "filler");
        try Io.Dir.cwd().writeFile(io, .{ .sub_path = try p.path("adapter.bin"), .data = "weights" });

        const before = try p.profileBytes();
        const per = before / 10;
        // Already under the target: nothing happens.
        try std.testing.expectEqual(@as(u32, 0), (try p.deleteOldestSamples(before)).deleted_samples);
        // Room for about four samples.
        const target = before - 6 * per;
        const rep = try p.deleteOldestSamples(target);
        try std.testing.expect(rep.deleted_samples >= 6 and rep.deleted_samples <= 7);
        try std.testing.expect(rep.bytes_after <= target);
        try std.testing.expectEqual(rep.bytes_after, try p.profileBytes());
        // Oldest gone, newest kept; vocabulary, rules, adapter untouched.
        try std.testing.expectError(error.FileNotFound, Io.Dir.cwd().statFile(io, try p.path(try std.fmt.bufPrint(&nb, "samples/{d:0>6}.wav", .{1})), .{}));
        _ = try Io.Dir.cwd().statFile(io, try p.path(try std.fmt.bufPrint(&nb, "samples/{d:0>6}.txt", .{10})), .{});
        _ = try Io.Dir.cwd().statFile(io, try p.path("adapter.bin"), .{});
        _ = try Io.Dir.cwd().statFile(io, try p.path("vocab.txt"), .{});
        try std.testing.expectEqual(@as(usize, 1), p.rules.items.len);

        // An interrupted delete: sample 9 lost only its .txt.
        try Io.Dir.cwd().deleteFile(io, try p.path(try std.fmt.bufPrint(&nb, "samples/{d:0>6}.txt", .{9})));
    }
    // Still loads, never reuses the leftover's id, and the next call
    // removes the leftover too.
    var p = try Profile.load(gpa, io, dir);
    defer p.deinit();
    try std.testing.expectEqual(@as(u32, 11), p.next_sample);
    try std.testing.expectEqual(@as(usize, 1), p.rules.items.len);
    _ = try p.deleteOldestSamples(0);
    try std.testing.expectError(error.FileNotFound, Io.Dir.cwd().statFile(io, try p.path(try std.fmt.bufPrint(&nb, "samples/{d:0>6}.wav", .{9})), .{}));
    _ = try Io.Dir.cwd().statFile(io, try p.path("adapter.bin"), .{});
    const left = try p.sampleIds(gpa);
    defer gpa.free(left);
    try std.testing.expectEqual(@as(usize, 0), left.len);
}
