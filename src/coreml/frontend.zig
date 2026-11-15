//! Log-mel spectrogram on the host CPU, for the Core ML backend (the GPU
//! and NPU engines compute it on their device). Same math as
//! gpu/kernels.zig (pad_audio, STFT by the windowed DFT basis, mel_power,
//! reduce_max, mel_norm):
//!
//!   zero-pad to 30 s, reflect-pad n_fft/2 = 200 on both sides,
//!   frame t = padded[t·160 .. t·160+400] (3000 frames),
//!   power = Re² + Im² of the 201-bin DFT (periodic Hann folded into the basis),
//!   mel = filters · power, log10(max(mel, 1e-10)),
//!   clamp to (max - 8), then (x + 4) / 4.
//!
//! Output layout [n_mels][3000] (what the encoder's first conv reads).
//! Frames that only see zero padding are not transformed (their row is
//! the floor), so a 5 s clip costs a sixth of a 30 s one. Single-threaded,
//! @Vector dot products: about 0.5 GFLOP for a full 30 s window.
const std = @import("std");
const mel = @import("../mel.zig");
const Model = @import("../model.zig").Model;

pub const n_bins = mel.n_fft / 2 + 1; // 201
pub const n_frames = mel.n_frames; // 3000
const pad = mel.n_fft / 2;
const floor_log10: f32 = -10.0; // log10(1e-10)

/// Computes the normalized log-mel spectrogram of up to 30 s of 16 kHz
/// audio into `out` ([n_mels][3000]).
pub fn logMel(m: *const Model, samples_in: []const f32, out: []f32) void {
    const n_mels = m.hp.n_mels;
    std.debug.assert(out.len == n_mels * n_frames);
    const samples = samples_in[0..@min(samples_in.len, mel.n_samples)];
    const n = samples.len;
    const total = mel.n_samples;
    const basis = mel.dftBasis();
    var frame: [mel.n_fft]f32 = undefined;
    var power: [n_bins]f32 = undefined;
    var mx: f32 = floor_log10;
    for (0..n_frames) |t| {
        var any = false;
        for (&frame, 0..) |*f, k| {
            const idx = t * mel.hop + k;
            const j: usize = if (idx < pad) pad - idx else if (idx >= pad + total) 2 * total + pad - 2 - idx else idx - pad;
            f.* = if (j < n) samples[j] else 0;
            any = any or f.* != 0;
        }
        if (!any) {
            for (0..n_mels) |r| out[r * n_frames + t] = floor_log10;
            continue;
        }
        for (0..n_bins) |k| {
            const re = dot(&basis.cos[k], &frame);
            const im = dot(&basis.sin[k], &frame);
            power[k] = re * re + im * im;
        }
        for (0..n_mels) |r| {
            const fr = m.mel_filters[r * n_bins ..][0..n_bins];
            var s: f32 = 0;
            for (fr, &power) |w, p| s += w * p;
            const v = @log10(@max(s, 1e-10));
            out[r * n_frames + t] = v;
            mx = @max(mx, v);
        }
    }
    for (out) |*v| v.* = (@max(v.*, mx - 8.0) + 4.0) / 4.0;
}

fn dot(a: *const [mel.n_fft]f32, b: *const [mel.n_fft]f32) f32 {
    const L = 8;
    const Vec = @Vector(L, f32);
    var s: Vec = @splat(0);
    var i: usize = 0;
    while (i + L <= mel.n_fft) : (i += L) s += @as(Vec, a[i..][0..L].*) * @as(Vec, b[i..][0..L].*);
    var r = @reduce(.Add, s);
    while (i < mel.n_fft) : (i += 1) r += a[i] * b[i];
    return r;
}

test "silence gives the floor everywhere, a tone lights up its bins" {
    const gpa = std.testing.allocator;
    // A fake model with a 2-mel filterbank: mel 0 sums bins 0..9, mel 1 bins 40..60.
    const filters = try gpa.alloc(f32, 2 * n_bins);
    defer gpa.free(filters);
    @memset(filters, 0);
    for (0..10) |k| filters[k] = 1;
    for (40..61) |k| filters[n_bins + k] = 1;
    var m: Model = undefined;
    m.hp.n_mels = 2;
    m.mel_filters = filters;
    const out = try gpa.alloc(f32, 2 * n_frames);
    defer gpa.free(out);
    logMel(&m, &.{}, out);
    // all floor: max = -10, so (max(-10, -18) + 4) / 4 = -1.5
    for (out) |v| try std.testing.expectEqual(@as(f32, -1.5), v);

    // 2 kHz tone for 1 s: bin 2000 / 40 Hz = 50
    const tone = try gpa.alloc(f32, 16000);
    defer gpa.free(tone);
    for (tone, 0..) |*s, i| s.* = 0.5 * @sin(2.0 * std.math.pi * 2000.0 * @as(f32, @floatFromInt(i)) / 16000.0);
    logMel(&m, tone, out);
    const mid = 50; // a frame inside the tone
    try std.testing.expect(out[n_frames + mid] > out[mid] + 1.0); // mel 1 (tone) above mel 0
    try std.testing.expectEqual(@as(f32, (-10.0 + 4.0) / 4.0) < out[n_frames + mid], true);
    // past the audio (frame 200 = 2 s): clamped floor = (max - 8 + 4) / 4
    var top: f32 = -std.math.inf(f32);
    for (out) |v| top = @max(top, v);
    const mx = top * 4.0 - 4.0;
    try std.testing.expectApproxEqAbs((@max(floor_log10, mx - 8.0) + 4.0) / 4.0, out[n_frames + 200], 1e-5);
}
