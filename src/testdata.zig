//! Test audio. The repository ships no audio files (THIRD_PARTY.md, Test
//! data): tests that only need some audio use `synth` / `synthWav`, which
//! generate it at run time, and tests that need real speech read an
//! optional, git-ignored recording at `speech_path` and skip
//! (error.SkipZigTest) when it is absent, as they do without the model.
const std = @import("std");
const Allocator = std.mem.Allocator;
const audio = @import("audio.zig");

/// Optional speech sample for the transcription tests: a short English
/// recording of `speech_text` (any WAV / MP3 / QOA; git-ignored, never
/// committed). Relative to the repository root, like models/.
pub const speech_path = "testdata/speech.wav";
/// What the speech sample says. The tests look for "Hello" in the
/// transcript and use the whole sentence as the corrected text.
pub const speech_text = "Hello world. This is a test of speech recognition written in pure Zig.";

/// Seed of the generated noise: the same samples on every run and machine.
pub const seed: u64 = 0x5354_545f_7465_7374;

/// `seconds` of generated audio at 16 kHz mono (not speech): a 300 Hz to
/// 3 kHz linear sine sweep at 0.3 plus seeded white noise at 0.05, faded in
/// and out over 20 ms. Caller owns the result.
pub fn synth(gpa: Allocator, seconds: f32) ![]f32 {
    const rate: f64 = audio.target_rate;
    const n: usize = @intFromFloat(seconds * audio.target_rate);
    const out = try gpa.alloc(f32, n);
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    const f0 = 300.0;
    const f1 = 3000.0;
    const dur = @as(f64, @floatFromInt(n)) / rate;
    const fade: usize = audio.target_rate / 50;
    for (out, 0..) |*o, i| {
        const t = @as(f64, @floatFromInt(i)) / rate;
        const phase = 2 * std.math.pi * (f0 * t + (f1 - f0) * t * t / (2 * dur));
        var v: f32 = 0.3 * @as(f32, @floatCast(@sin(phase))) + 0.05 * (2 * rand.float(f32) - 1);
        const edge = @min(i, n - 1 - i);
        if (edge < fade) v *= @as(f32, @floatFromInt(edge)) / @as(f32, @floatFromInt(fade));
        o.* = v;
    }
    return out;
}

/// `synth` as a 16 kHz mono 16-bit PCM WAV file (`audio.encodeWav`).
/// Caller owns the result.
pub fn synthWav(gpa: Allocator, seconds: f32) ![]u8 {
    const samples = try synth(gpa, seconds);
    defer gpa.free(samples);
    return audio.encodeWav(gpa, samples, audio.target_rate);
}

/// The speech sample's file bytes, or error.SkipZigTest when there is none.
/// Caller owns the result.
pub fn speechFile(gpa: Allocator, io: std.Io) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, speech_path, gpa, .limited(16 << 20)) catch |err| switch (err) {
        error.FileNotFound => error.SkipZigTest,
        else => err,
    };
}

/// The speech sample decoded to 16 kHz mono f32, or error.SkipZigTest when
/// there is none. Caller owns the result.
pub fn speechSamples(gpa: Allocator, io: std.Io) ![]f32 {
    const bytes = try speechFile(gpa, io);
    defer gpa.free(bytes);
    return audio.decode(gpa, bytes, .{});
}

test "synth is deterministic and survives a WAV round trip" {
    const gpa = std.testing.allocator;
    const a = try synth(gpa, 0.5);
    defer gpa.free(a);
    const b = try synth(gpa, 0.5);
    defer gpa.free(b);
    try std.testing.expectEqual(@as(usize, audio.target_rate / 2), a.len);
    try std.testing.expectEqualSlices(f32, a, b);
    try std.testing.expectEqual(@as(f32, 0), a[0]);
    for (a) |v| try std.testing.expect(@abs(v) <= 0.35);

    const wav = try synthWav(gpa, 0.5);
    defer gpa.free(wav);
    try std.testing.expectEqualStrings("RIFF", wav[0..4]);
    const back = try audio.decode(gpa, wav, .{});
    defer gpa.free(back);
    try std.testing.expectEqual(a.len, back.len);
    for (a, back) |x, y| try std.testing.expectApproxEqAbs(x, y, 1e-3);
}
