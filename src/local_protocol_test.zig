//! End-to-end tests of the `stt-local` JSON-lines protocol: they spawn the
//! built executable on the cuda backend (need models/ggml-tiny.en.bin and an
//! NVIDIA GPU; skipped otherwise). The framing test sends generated audio
//! (stt.testdata.synthWav); the transcription test needs real speech, the
//! optional speech sample (stt.testdata.speech_path, git-ignored), and is
//! skipped without it.
const std = @import("std");
const Io = std.Io;
const stt = @import("stt");
const testdata = stt.testdata;
const exe_path = @import("stt_local_build").exe_path;

const model_path = "models/ggml-tiny.en.bin";

const Line = std.json.Parsed(std.json.Value);

fn field(v: std.json.Value, key: []const u8) ?std.json.Value {
    if (v != .object) return null;
    return v.object.get(key);
}

fn fieldStr(v: std.json.Value, key: []const u8) []const u8 {
    const f = field(v, key) orelse return "";
    return if (f == .string) f.string else "";
}

fn fieldInt(v: std.json.Value, key: []const u8) ?i64 {
    const f = field(v, key) orelse return null;
    return if (f == .integer) f.integer else null;
}

/// Next non-event line (events are checked for shape and skipped).
fn nextReply(gpa: std.mem.Allocator, r: *Io.Reader) !Line {
    while (true) {
        const raw = (try r.takeDelimiter('\n')) orelse return error.EndOfStream;
        const p = try std.json.parseFromSlice(std.json.Value, gpa, raw, .{});
        if (field(p.value, "event") != null) {
            try std.testing.expectEqualStrings("wakeup", fieldStr(p.value, "event"));
            try std.testing.expect(field(p.value, "deadline_unix_ms") != null);
            p.deinit();
            continue;
        }
        return p;
    }
}

fn b64(gpa: std.mem.Allocator, bytes: []const u8) ![]u8 {
    const enc = std.base64.standard.Encoder;
    const out = try gpa.alloc(u8, enc.calcSize(bytes.len));
    _ = enc.encode(out, bytes);
    return out;
}

/// The 16 kHz mono s16le samples of `wav` (any WAV / MP3 / QOA) for a
/// `pcm_s16le_b64` request: decoded, then re-encoded by stt.audio.encodeWav,
/// whose data chunk is the PCM. Caller owns the result.
fn pcmOf(gpa: std.mem.Allocator, wav: []const u8) ![]u8 {
    const samples = try stt.audio.decode(gpa, wav, .{});
    defer gpa.free(samples);
    const canon = try stt.audio.encodeWav(gpa, samples, stt.audio.target_rate);
    defer gpa.free(canon);
    const at = std.mem.indexOf(u8, canon, "data").? + 8;
    return gpa.dupe(u8, canon[at..]);
}

/// Sends `wav` as raw PCM (rid 13) and as a file (rid 14).
fn sendAudio(gpa: std.mem.Allocator, w: *Io.Writer, wav: []const u8) !void {
    const pcm = try pcmOf(gpa, wav);
    defer gpa.free(pcm);
    const pcm_enc = try b64(gpa, pcm);
    defer gpa.free(pcm_enc);
    try w.print("{{\"rid\":13,\"op\":\"transcribe\",\"pcm_s16le_b64\":\"{s}\"}}\n", .{pcm_enc});
    const enc = try b64(gpa, wav);
    defer gpa.free(enc);
    try w.print("{{\"rid\":14,\"op\":\"transcribe\",\"wav_b64\":\"{s}\"}}\n", .{enc});
}

/// Runs stt-local (cuda backend, fresh profile), checks its ready event,
/// runs `body` with its stdout reader and stdin writer, then closes stdin:
/// the program must exit cleanly within 2 s. Skips without the model or a GPU.
fn session(wav: []const u8, comptime body: fn (std.mem.Allocator, []const u8, *Io.Reader, *Io.Writer) anyerror!void) !void {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    Io.Dir.cwd().access(io, model_path, .{}) catch return error.SkipZigTest;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);

    var child = try std.process.spawn(io, .{
        .argv = &.{ exe_path, "--model", model_path, "--profile", dir, "--backend", "cuda" },
        .stdin = .pipe,
        .stdout = .pipe,
        .stderr = .inherit,
        .create_no_window = true,
    });
    var waited = false;
    defer if (!waited) child.kill(io);

    var out_buf: [1 << 16]u8 = undefined;
    var out_r = Io.File.Reader.initStreaming(child.stdout.?, io, &out_buf);
    const r = &out_r.interface;
    var in_buf: [1 << 16]u8 = undefined;
    var in_w = Io.File.Writer.initStreaming(child.stdin.?, io, &in_buf);
    const w = &in_w.interface;

    // The first line is the ready event.
    const first = (try r.takeDelimiter('\n')) orelse {
        // No GPU: stt-local exits before ready.
        _ = child.wait(io) catch {};
        waited = true;
        return error.SkipZigTest;
    };
    {
        const p = try std.json.parseFromSlice(std.json.Value, gpa, first, .{});
        defer p.deinit();
        try std.testing.expectEqualStrings("ready", fieldStr(p.value, "event"));
        try std.testing.expectEqual(@as(?i64, 1), fieldInt(p.value, "proto"));
        try std.testing.expectEqualStrings("cuda", fieldStr(p.value, "backend"));
        try std.testing.expectEqualStrings("ggml-tiny.en.bin", fieldStr(p.value, "model"));
        try std.testing.expectEqual(@as(?i64, 0), fieldInt(p.value, "adapter_version"));
        try std.testing.expect(fieldStr(p.value, "version").len > 0);
    }

    try body(gpa, wav, r, w);

    // Closing stdin stops the program; it must exit cleanly within 2 s.
    w.flush() catch {};
    child.stdin.?.close(io);
    child.stdin = null;
    const t0 = Io.Clock.awake.now(io);
    while (try r.takeDelimiter('\n')) |raw| {
        const p = try std.json.parseFromSlice(std.json.Value, gpa, raw, .{});
        defer p.deinit();
        try std.testing.expect(field(p.value, "event") != null); // only events after the last reply
    }
    const term = try child.wait(io);
    waited = true;
    const ms = t0.durationTo(Io.Clock.awake.now(io)).toMilliseconds();
    try std.testing.expect(term == .exited and term.exited == 0);
    try std.testing.expect(ms < 2000);
}

test "stt-local protocol: ready, rid echo, error codes, audio framing, clean exit on EOF" {
    const gpa = std.testing.allocator;
    // Generated audio (not speech): only the framing is checked here.
    const wav = try testdata.synthWav(gpa, 3);
    defer gpa.free(wav);
    try session(wav, framing);
}

fn framing(gpa: std.mem.Allocator, wav: []const u8, r: *Io.Reader, w: *Io.Writer) !void {
    // Malformed JSON: bad_request with rid 0.
    try w.writeAll("this is not json\n");
    // Bad audio.
    {
        const junk = try b64(gpa, "definitely not a wav file");
        defer gpa.free(junk);
        try w.print("{{\"rid\":11,\"op\":\"transcribe\",\"wav_b64\":\"{s}\"}}\n", .{junk});
    }
    // Too long: 121 s of s16le silence.
    {
        const zeros = try gpa.alloc(u8, 121 * 16000 * 2 + 4);
        defer gpa.free(zeros);
        @memset(zeros, 0);
        const enc = try b64(gpa, zeros);
        defer gpa.free(enc);
        try w.print("{{\"rid\":12,\"op\":\"transcribe\",\"pcm_s16le_b64\":\"{s}\"}}\n", .{enc});
    }
    // Audio as raw PCM and as a WAV file: transcribed (whatever the text).
    try sendAudio(gpa, w, wav);
    try w.writeAll("{\"rid\":15,\"op\":\"feedback\",\"id\":999999,\"text\":\"x\",\"ok\":true}\n");
    try w.writeAll("{\"rid\":16,\"op\":\"nope\"}\n");
    try w.writeAll("{\"rid\":17,\"op\":\"status\"}\n");
    try w.flush();

    const Expect = struct { rid: i64, err: ?[]const u8 };
    const expected = [_]Expect{
        .{ .rid = 0, .err = "bad_request" },
        .{ .rid = 11, .err = "bad_audio" },
        .{ .rid = 12, .err = "too_long" },
        .{ .rid = 13, .err = null },
        .{ .rid = 14, .err = null },
        .{ .rid = 15, .err = "unknown_id" },
        .{ .rid = 16, .err = "bad_request" },
        .{ .rid = 17, .err = null },
    };
    for (expected) |e| {
        const p = try nextReply(gpa, r);
        defer p.deinit();
        try std.testing.expectEqual(@as(?i64, e.rid), fieldInt(p.value, "rid"));
        if (e.err) |code| {
            try std.testing.expectEqualStrings(code, fieldStr(p.value, "error"));
            try std.testing.expect(field(p.value, "message") != null);
        } else {
            try std.testing.expect(field(p.value, "error") == null);
        }
        if (e.rid == 13 or e.rid == 14) {
            try std.testing.expect(field(p.value, "text").? == .string);
            try std.testing.expect(fieldInt(p.value, "id").? >= 0);
        }
        if (e.rid == 17) {
            try std.testing.expectEqual(@as(?i64, 1), fieldInt(p.value, "proto"));
            try std.testing.expectEqualStrings("cuda", fieldStr(p.value, "backend"));
        }
    }
}

test "stt-local protocol: transcribes speech (PCM and file) and learns a fix" {
    const gpa = std.testing.allocator;
    // Needs real speech: skipped without the speech sample, as without the model.
    const wav = try testdata.speechFile(gpa, std.testing.io);
    defer gpa.free(wav);
    try session(wav, speech);
}

fn speech(gpa: std.mem.Allocator, wav: []const u8, r: *Io.Reader, w: *Io.Writer) !void {
    try sendAudio(gpa, w, wav);
    try w.flush();
    var id14: i64 = 0;
    for ([_]i64{ 13, 14 }) |rid| {
        const p = try nextReply(gpa, r);
        defer p.deinit();
        try std.testing.expectEqual(@as(?i64, rid), fieldInt(p.value, "rid"));
        try std.testing.expect(field(p.value, "error") == null);
        try std.testing.expect(std.mem.indexOf(u8, fieldStr(p.value, "text"), "Hello") != null);
        try std.testing.expect(fieldInt(p.value, "id").? > 0);
        if (rid == 14) id14 = fieldInt(p.value, "id").?;
    }

    // A fix for the WAV transcript is learned.
    try w.print("{{\"rid\":18,\"op\":\"feedback\",\"id\":{d},\"text\":\"{s}\",\"ok\":true}}\n", .{ id14, testdata.speech_text });
    try w.flush();
    const p = try nextReply(gpa, r);
    defer p.deinit();
    try std.testing.expectEqual(@as(?i64, 18), fieldInt(p.value, "rid"));
    try std.testing.expectEqualStrings("learned", fieldStr(p.value, "outcome"));
}
