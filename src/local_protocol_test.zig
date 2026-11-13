//! End-to-end test of the `stt-local` JSON-lines protocol: spawns the built
//! executable on the cuda backend (needs models/ggml-tiny.en.bin,
//! testdata/hello.wav and an NVIDIA GPU; skipped otherwise).
const std = @import("std");
const Io = std.Io;
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

test "stt-local protocol: ready, rid echo, error codes, transcribe, clean exit on EOF" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const cwd = Io.Dir.cwd();
    cwd.access(io, model_path, .{}) catch return error.SkipZigTest;
    const wav = cwd.readFileAlloc(io, "testdata/hello.wav", gpa, .limited(16 << 20)) catch return error.SkipZigTest;
    defer gpa.free(wav);

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

    // 1. The first line is the ready event.
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

    // 2. Malformed JSON: bad_request with rid 0.
    try w.writeAll("this is not json\n");
    // 3. Bad audio.
    {
        const junk = try b64(gpa, "definitely not a wav file");
        defer gpa.free(junk);
        try w.print("{{\"rid\":11,\"op\":\"transcribe\",\"wav_b64\":\"{s}\"}}\n", .{junk});
    }
    // 4. Too long: 121 s of s16le silence.
    {
        const zeros = try gpa.alloc(u8, 121 * 16000 * 2 + 4);
        defer gpa.free(zeros);
        @memset(zeros, 0);
        const enc = try b64(gpa, zeros);
        defer gpa.free(enc);
        try w.print("{{\"rid\":12,\"op\":\"transcribe\",\"pcm_s16le_b64\":\"{s}\"}}\n", .{enc});
    }
    // 5. Real transcriptions: raw PCM (hello.wav is 16 kHz mono s16) and the WAV file.
    {
        const at = std.mem.indexOf(u8, wav, "data").? + 8;
        const pcm = try b64(gpa, wav[at..]);
        defer gpa.free(pcm);
        try w.print("{{\"rid\":13,\"op\":\"transcribe\",\"pcm_s16le_b64\":\"{s}\"}}\n", .{pcm});
        const enc = try b64(gpa, wav);
        defer gpa.free(enc);
        try w.print("{{\"rid\":14,\"op\":\"transcribe\",\"wav_b64\":\"{s}\"}}\n", .{enc});
    }
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
    var id14: i64 = 0;
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
            try std.testing.expect(std.mem.indexOf(u8, fieldStr(p.value, "text"), "Hello") != null);
            try std.testing.expect(fieldInt(p.value, "id").? > 0);
            if (e.rid == 14) id14 = fieldInt(p.value, "id").?;
        }
        if (e.rid == 17) {
            try std.testing.expectEqual(@as(?i64, 1), fieldInt(p.value, "proto"));
            try std.testing.expectEqualStrings("cuda", fieldStr(p.value, "backend"));
        }
    }

    // A fix for the WAV transcript is learned.
    try w.print("{{\"rid\":18,\"op\":\"feedback\",\"id\":{d},\"text\":\"Hello world. This is a test of speech recognition written in pure Zig.\",\"ok\":true}}\n", .{id14});
    try w.flush();
    {
        const p = try nextReply(gpa, r);
        defer p.deinit();
        try std.testing.expectEqual(@as(?i64, 18), fieldInt(p.value, "rid"));
        try std.testing.expectEqualStrings("learned", fieldStr(p.value, "outcome"));
    }

    // Closing stdin stops the program; it must exit cleanly within 2 s.
    in_w.interface.flush() catch {};
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
