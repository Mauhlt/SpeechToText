//! Audio in and out, via the `Audio` library (../Audio): WAV, MP3 and QOA,
//! any sample type, rate or channel count. This file only adapts it to what
//! the model wants: 16 kHz mono f32.
const std = @import("std");
const Allocator = std.mem.Allocator;
const Audio = @import("Audio");

pub const target_rate = 16000;

pub const Limits = struct {
    /// Reject audio longer than this before decoding it (memory guard for
    /// untrusted input from Portal / sync bundles).
    max_seconds: u32 = 120,
};

/// Decodes WAV / MP3 / QOA bytes (format detected from the header) to 16 kHz
/// mono f32. Headers are checked against `limits` before any sample is
/// decoded. Caller owns the result.
pub fn decode(gpa: Allocator, bytes: []const u8, limits: Limits) ![]f32 {
    const tag = Audio.tagFromMagic(bytes) orelse return error.UnknownAudioFormat;
    if (!Audio.canDecode(tag)) return error.UnsupportedAudioFormat;
    const inf = try Audio.info(tag, bytes);
    if (inf.channels == 0 or inf.channels > 32) return error.BadAudio;
    if (inf.sample_rate < 4000 or inf.sample_rate > 384000) return error.UnsupportedSampleRate;
    if (inf.seconds() > @as(f64, @floatFromInt(limits.max_seconds))) return error.AudioTooLong;

    const decoded = try Audio.decode(tag, gpa, bytes);
    defer decoded.deinit(gpa);
    const f = try decoded.convert(.f32, gpa);
    defer f.deinit(gpa);
    const mono = try downmix(gpa, f.samples.f32, decoded.channels);
    defer gpa.free(mono);
    return resample(gpa, mono, decoded.sample_rate, target_rate);
}

/// Reads and decodes an audio file (see `decode`).
pub fn load(gpa: Allocator, io: std.Io, path: []const u8, limits: Limits) ![]f32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(256 << 20));
    defer gpa.free(bytes);
    return decode(gpa, bytes, limits);
}

/// Encodes mono f32 samples at `rate` as a 16-bit PCM WAV file. Caller owns the result.
pub fn encodeWav(gpa: Allocator, samples: []const f32, rate: u32) ![]u8 {
    const a: Audio = .{ .sample_rate = rate, .channels = 1, .samples = .{ .f32 = @constCast(samples) } };
    const pcm = try a.convert(.i16, gpa);
    defer pcm.deinit(gpa);
    var aw: std.Io.Writer.Allocating = .init(gpa);
    errdefer aw.deinit();
    try pcm.encode(.wav, &aw.writer, gpa);
    return aw.toOwnedSlice();
}

fn downmix(gpa: Allocator, interleaved: []const f32, channels: u16) ![]f32 {
    if (channels == 1) return gpa.dupe(f32, interleaved);
    const n = interleaved.len / channels;
    const out = try gpa.alloc(f32, n);
    const inv = 1.0 / @as(f32, @floatFromInt(channels));
    for (out, 0..) |*o, i| {
        var s: f32 = 0;
        for (interleaved[i * channels ..][0..channels]) |v| s += v;
        o.* = s * inv;
    }
    return out;
}

/// Linear-interpolation resampler. Returns a new buffer owned by the caller.
pub fn resample(gpa: Allocator, in: []const f32, from: u32, to: u32) ![]f32 {
    if (from == to or in.len == 0) return gpa.dupe(f32, in);
    const n_out: usize = @intCast(@as(u64, in.len) * to / from);
    const out = try gpa.alloc(f32, n_out);
    const ratio = @as(f64, @floatFromInt(from)) / @as(f64, @floatFromInt(to));
    for (out, 0..) |*o, i| {
        const p = @as(f64, @floatFromInt(i)) * ratio;
        const j: usize = @intFromFloat(p);
        const f: f32 = @floatCast(p - @as(f64, @floatFromInt(j)));
        const a = in[@min(j, in.len - 1)];
        const b = in[@min(j + 1, in.len - 1)];
        o.* = a + (b - a) * f;
    }
    return out;
}

test "wav round trip and limits" {
    const gpa = std.testing.allocator;
    var tone: [16000]f32 = undefined;
    for (&tone, 0..) |*v, i| v.* = 0.5 * @sin(@as(f32, @floatFromInt(i)) * 0.05);
    const bytes = try encodeWav(gpa, &tone, target_rate);
    defer gpa.free(bytes);
    const back = try decode(gpa, bytes, .{});
    defer gpa.free(back);
    try std.testing.expectEqual(tone.len, back.len);
    for (tone, back) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-3);
    try std.testing.expectError(error.AudioTooLong, decode(gpa, bytes, .{ .max_seconds = 0 }));
    try std.testing.expectError(error.UnknownAudioFormat, decode(gpa, "not audio", .{}));
}
