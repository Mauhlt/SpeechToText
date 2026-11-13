//! Constants shared by the GPU kernels (device code) and the engine (host code).

/// Attention head size. Every supported model size uses 64.
pub const head_dim = 64;
/// Keys per block in single-query (decoder) attention.
pub const split = 128;
/// Partial result per (head, split): max, sum, then the head_dim outputs.
pub const part_stride = head_dim + 2;
/// STFT size of the log-mel front end.
pub const n_fft = 400;
pub const n_bins = n_fft / 2 + 1;

// Epilogue flags for gemm/gemv: out = [accum ? out : 0] + act(acc + bias).
pub const flag_bias: u32 = 1;
pub const flag_gelu: u32 = 2;
pub const flag_accum: u32 = 4;
