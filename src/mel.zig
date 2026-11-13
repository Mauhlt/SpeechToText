//! Log-mel spectrogram front end parameters. The spectrogram itself is
//! computed on the GPU (gpu/kernels.zig: pad_audio, STFT via gemm,
//! mel_power, mel_norm):
//! 16 kHz audio, padded/trimmed to 30 s, STFT (n_fft=400, hop=160, periodic
//! Hann window, centered with reflect padding), power spectrum, mel filterbank,
//! log10, dynamic-range clamp to (max - 8), then (x + 4) / 4.
const std = @import("std");

pub const sample_rate = 16000;
pub const n_fft = 400;
pub const hop = 160;
pub const chunk_seconds = 30;
pub const n_samples = sample_rate * chunk_seconds; // 480000
pub const n_frames = n_samples / hop; // 3000
const n_bins = n_fft / 2 + 1; // 201

/// Precomputed DFT basis (window folded in): rows are bins, columns samples.
pub const Basis = extern struct {
    cos: [n_bins][n_fft]f32,
    sin: [n_bins][n_fft]f32,
};

var basis: Basis = undefined;
var basis_ready: std.atomic.Value(bool) = .init(false);
var basis_lock: std.atomic.Mutex = .unlocked;

/// The windowed DFT basis, as [2 * 201][400] (cos rows, then sin rows),
/// uploaded to the GPU once per engine.
pub fn dftBasis() *const Basis {
    if (basis_ready.load(.acquire)) return &basis;
    // Inference and training engines may start on different threads at once.
    while (!basis_lock.tryLock()) std.atomic.spinLoopHint();
    defer basis_lock.unlock();
    if (!basis_ready.load(.acquire)) {
        for (0..n_bins) |k| {
            for (0..n_fft) |n| {
                const win = 0.5 * (1.0 - @cos(2.0 * std.math.pi * @as(f64, @floatFromInt(n)) / n_fft));
                const ang = 2.0 * std.math.pi * @as(f64, @floatFromInt((k * n) % n_fft)) / n_fft;
                basis.cos[k][n] = @floatCast(win * @cos(ang));
                basis.sin[k][n] = @floatCast(win * @sin(ang));
            }
        }
        basis_ready.store(true, .release);
    }
    return &basis;
}
