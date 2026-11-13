//! Transcription + feedback on the home device, for clients that send audio
//! through Portal (e.g. a phone). The home GPU transcribes it, returns
//! `{ id, text }`, and later feedback trains the home profile directly,
//! tagged with the client's device name. Most hosts use `Home`, which wraps
//! this together with scheduled training.
//!
//! Thread-safe: requests are serialized (one GPU).
const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const Transcriber = @import("transcriber.zig").Transcriber;
const Profile = @import("profile.zig").Profile;
const window = @import("window.zig");
const audio = @import("audio.zig");

pub const Outcome = window.Outcome;

pub const Options = struct {
    /// Transcripts that can still receive feedback (shared by all clients).
    feedback_window: usize = 64,
    /// Silence = correct after this long with no feedback.
    confirm_after_ms: i64 = 60_000,
    /// Requests with more audio than this are rejected (memory guard).
    max_seconds: u32 = 120,
};

pub const Result = struct {
    id: u64,
    /// Owned by the caller; free with the allocator passed to `transcribe*`.
    text: []u8,
};

pub const Service = struct {
    gpa: Allocator,
    io: Io,
    tr: *Transcriber,
    profile: ?*Profile,
    opts: Options,
    win: window.Window,
    lock: Io.Mutex = .init,

    pub fn init(gpa: Allocator, io: Io, tr: *Transcriber, profile: ?*Profile, opts: Options) !Service {
        return .{
            .gpa = gpa,
            .io = io,
            .tr = tr,
            .profile = profile,
            .opts = opts,
            .win = try .init(gpa, io, profile, opts.feedback_window, opts.confirm_after_ms),
        };
    }

    /// Confirms whatever is still pending (silence = correct), then frees.
    pub fn deinit(s: *Service) void {
        s.win.confirmAll();
        s.win.deinit();
    }

    /// Transcribes an audio file's bytes: WAV, MP3 or QOA, any sample rate or
    /// channel count. Untrusted input is validated (length is checked from
    /// the header before decoding).
    pub fn transcribeAudio(s: *Service, gpa: Allocator, audio_bytes: []const u8, device: []const u8) !Result {
        const samples = audio.decode(gpa, audio_bytes, .{ .max_seconds = s.opts.max_seconds }) catch |err| return switch (err) {
            error.AudioTooLong, error.OutOfMemory => err,
            else => error.BadAudio,
        };
        defer gpa.free(samples);
        return s.transcribe(gpa, samples, device);
    }

    /// Transcribes 16 kHz mono f32 samples sent by `device`.
    pub fn transcribe(s: *Service, gpa: Allocator, samples: []const f32, device: []const u8) !Result {
        if (samples.len > @as(usize, s.opts.max_seconds) * audio.target_rate) return error.AudioTooLong;
        s.lock.lockUncancelable(s.io);
        defer s.lock.unlock(s.io);
        s.win.confirmExpired();

        const raw = try s.tr.transcribe(gpa, samples);
        defer gpa.free(raw);
        const text = if (s.profile) |p| try p.applyRules(gpa, raw) else try gpa.dupe(u8, raw);
        errdefer gpa.free(text);
        // Empty results get id 0: there is nothing to give feedback on.
        if (text.len == 0) return .{ .id = 0, .text = text };
        const id = try s.win.add(samples, text, device);
        return .{ .id = id, .text = text };
    }

    /// The client's response for transcript `id` (see `window` for meaning).
    pub fn feedback(s: *Service, id: u64, text: []const u8, success: bool) !Outcome {
        s.lock.lockUncancelable(s.io);
        defer s.lock.unlock(s.io);
        const out = s.win.feedback(id, text, success);
        if (out == .learned and out.learned.new_words > 0) try s.refreshPrompt();
        return out;
    }

    /// Confirms transcripts whose silence period has passed. Call it at
    /// `msUntilTick` (or periodically).
    pub fn tick(s: *Service) void {
        s.lock.lockUncancelable(s.io);
        defer s.lock.unlock(s.io);
        s.win.confirmExpired();
    }

    /// Milliseconds until `tick` next has work (a silent transcript to
    /// confirm), or null if nothing is pending.
    pub fn msUntilTick(s: *Service) ?i64 {
        s.lock.lockUncancelable(s.io);
        defer s.lock.unlock(s.io);
        return s.win.msUntilNextConfirm();
    }

    fn refreshPrompt(s: *Service) !void {
        const p = s.profile orelse return;
        const text = try p.promptText(s.gpa, "");
        defer s.gpa.free(text);
        try s.tr.setPrompt(text);
    }
};

test "service: transcribe for a thin client, then learn from its fix" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const Model = @import("model.zig").Model;
    var model = Model.load(gpa, io, "models/ggml-tiny.en.bin") catch return error.SkipZigTest;
    defer model.deinit();
    const wav_bytes = Io.Dir.cwd().readFileAlloc(io, "testdata/hello.wav", gpa, .limited(16 << 20)) catch return error.SkipZigTest;
    defer gpa.free(wav_bytes);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    var home = try Profile.load(gpa, io, dir);
    defer home.deinit();
    var tr = try Transcriber.init(gpa, &model, .{});
    defer tr.deinit();
    var svc = try Service.init(gpa, io, &tr, &home, .{});
    defer svc.deinit();

    const r = try svc.transcribeAudio(gpa, wav_bytes, "iphone");
    defer gpa.free(r.text);
    try std.testing.expect(r.id != 0);
    try std.testing.expect(std.mem.indexOf(u8, r.text, "Hello") != null);

    const out = try svc.feedback(r.id, "Hello world. This is a test of speech recognition written in pure Zig.", true);
    try std.testing.expect(out == .learned);
    try std.testing.expect(try svc.feedback(r.id, "again", true) == .unknown_id);
    try std.testing.expectError(error.BadAudio, svc.transcribeAudio(gpa, "not audio", "iphone"));

    // The sample is tagged with the thin client's name.
    var nb: [48]u8 = undefined;
    const meta = try Io.Dir.cwd().readFileAlloc(io, try home.path(try std.fmt.bufPrint(&nb, "samples/{d:0>6}.meta", .{out.learned.sample_id.?})), gpa, .limited(4096));
    defer gpa.free(meta);
    try std.testing.expect(std.mem.indexOf(u8, meta, "source=iphone") != null);
}
