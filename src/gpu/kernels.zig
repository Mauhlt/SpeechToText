//! GPU kernels for the transformer, written in Zig and compiled by Zig's
//! NVPTX backend to PTX text (see build.zig). The driver JIT-compiles that PTX
//! for whatever NVIDIA GPU is installed, so no CUDA toolkit is involved.
//!
//! Conventions: activations are f32; weight matrices are f16 (half the VRAM and
//! half the memory traffic, and the weight files are f16 to begin with);
//! accumulation is always f32. Head dimension is fixed at 64 (true for every
//! model size this library loads). Inference kernels come first, then the
//! training ones (LoRA backward pass and optimizers).
const std = @import("std");

const F32 = [*]addrspace(.global) f32;
const CF32 = [*]addrspace(.global) const f32;
const CF16 = [*]addrspace(.global) const f16;
const U32 = [*]addrspace(.global) u32;
const shared = @import("shared.zig");

const head_dim = shared.head_dim;
const log2e: f32 = 1.4426950408889634;
const ln2: f32 = 0.6931471805599453;
const inf = std.math.inf(f32);

const flag_bias = shared.flag_bias;
const flag_gelu = shared.flag_gelu;
const flag_accum = shared.flag_accum;

// ---------------------------------------------------------------- intrinsics

inline fn tid() u32 {
    return asm ("mov.u32 \t%[r], %tid.x;"
        : [r] "=r" (-> u32),
    );
}
inline fn ctaX() u32 {
    return asm ("mov.u32 \t%[r], %ctaid.x;"
        : [r] "=r" (-> u32),
    );
}
inline fn ctaY() u32 {
    return asm ("mov.u32 \t%[r], %ctaid.y;"
        : [r] "=r" (-> u32),
    );
}
inline fn ctaZ() u32 {
    return asm ("mov.u32 \t%[r], %ctaid.z;"
        : [r] "=r" (-> u32),
    );
}
inline fn nctaX() u32 {
    return asm ("mov.u32 \t%[r], %nctaid.x;"
        : [r] "=r" (-> u32),
    );
}
inline fn ntid() u32 {
    return asm ("mov.u32 \t%[r], %ntid.x;"
        : [r] "=r" (-> u32),
    );
}
inline fn barrier() void {
    asm volatile ("bar.sync \t0;" ::: .{ .memory = true });
}
inline fn exp2(x: f32) f32 {
    return asm ("ex2.approx.ftz.f32 \t%[r], %[x];"
        : [r] "=f" (-> f32),
        : [x] "f" (x),
    );
}
inline fn log2(x: f32) f32 {
    return asm ("lg2.approx.ftz.f32 \t%[r], %[x];"
        : [r] "=f" (-> f32),
        : [x] "f" (x),
    );
}
inline fn shflXor(v: f32, off: u32) f32 {
    return asm volatile ("shfl.sync.bfly.b32 \t%[r], %[v], %[o], 0x1f, 0xffffffff;"
        : [r] "=f" (-> f32),
        : [v] "f" (v),
          [o] "r" (off),
    );
}
inline fn shflXorU(v: u32, off: u32) u32 {
    return asm volatile ("shfl.sync.bfly.b32 \t%[r], %[v], %[o], 0x1f, 0xffffffff;"
        : [r] "=r" (-> u32),
        : [v] "r" (v),
          [o] "r" (off),
    );
}
inline fn warpSum(v0: f32) f32 {
    var v = v0;
    inline for (.{ 16, 8, 4, 2, 1 }) |o| v += shflXor(v, o);
    return v;
}
inline fn warpMax(v0: f32) f32 {
    var v = v0;
    inline for (.{ 16, 8, 4, 2, 1 }) |o| v = @max(v, shflXor(v, o));
    return v;
}

/// tanh-approximation GELU, matching the CPU path. tanh(y) = 1 - 2/(e^{2y}+1).
inline fn gelu(u: f32) f32 {
    const y = 0.7978845608028654 * (u + 0.044715 * u * u * u);
    const th = 1.0 - 2.0 / (exp2(2.0 * log2e * y) + 1.0);
    return 0.5 * u * (1.0 + th);
}

var red: [32]f32 addrspace(.shared) = undefined;

/// Sum over the whole block (blockDim must be a multiple of 32). All threads get the result.
fn blockSum(v: f32) f32 {
    const w = warpSum(v);
    const t = tid();
    if (t % 32 == 0) red[t / 32] = w;
    barrier();
    var s: f32 = 0;
    for (0..ntid() / 32) |i| s += red[i];
    barrier();
    return s;
}
fn blockMax(v: f32) f32 {
    const w = warpMax(v);
    const t = tid();
    if (t % 32 == 0) red[t / 32] = w;
    barrier();
    var s: f32 = -inf;
    for (0..ntid() / 32) |i| s = @max(s, red[i]);
    barrier();
    return s;
}

inline fn epilogue(out: F32, idx: u32, acc: f32, bias: ?CF32, col: u32, flags: u32) void {
    var v = acc;
    if (flags & flag_bias != 0) v += bias.?[col];
    if (flags & flag_gelu != 0) v = gelu(v);
    if (flags & flag_accum != 0) v += out[idx];
    out[idx] = v;
}

// ---------------------------------------------------------------- matmul

const BM = 64;
const BN = 64;
const BK = 16;
var tile_a: [BK][BM + 4]f32 align(16) addrspace(.shared) = undefined;
var tile_w: [BK][BN + 4]f32 align(16) addrspace(.shared) = undefined;
const V4 = @Vector(4, f32);
/// 16-byte-vector views of the tiles, for ld.shared.v4 reads.
inline fn tileA() *addrspace(.shared) [BK][(BM + 4) / 4]V4 {
    return @ptrCast(&tile_a);
}
inline fn tileW() *addrspace(.shared) [BK][(BN + 4) / 4]V4 {
    return @ptrCast(&tile_w);
}

/// General batched matmul with arbitrary strides, used for every matrix
/// product in inference and training (A·Wᵀ, A·B, Aᵀ·B, per-head attention):
///
///   out[b][i][j] (=|+=) alpha · Σ_p A(b,i,p) · W(b,p,j)   (+ bias[j], gelu)
///   A(b,i,p) = a[b*sa_b + i*sa_i + p*sa_p]
///   W(b,p,j) = w[b*sw_b + p*sw_p + j*sw_j]
///   out      = out[b*so_b + i*so_i + j]
///
/// Grid (ceil(n/64), ceil(m/64), batch), 256 threads; each thread computes a
/// 4x4 block of outputs from shared-memory tiles. `gemm` takes f16 weights,
/// `gemm_f32` f32 ones.
export fn gemm(out: F32, so_i: u32, so_b: u32, a: CF32, sa_i: u32, sa_p: u32, sa_b: u32, w: CF16, sw_p: u32, sw_j: u32, sw_b: u32, bias: ?CF32, m: u32, n: u32, k: u32, flags: u32, alpha: f32) callconv(.nvptx_kernel) void {
    gemmT(f16, out, so_i, so_b, a, sa_i, sa_p, sa_b, w, sw_p, sw_j, sw_b, bias, m, n, k, flags, alpha);
}

export fn gemm_f32(out: F32, so_i: u32, so_b: u32, a: CF32, sa_i: u32, sa_p: u32, sa_b: u32, w: CF32, sw_p: u32, sw_j: u32, sw_b: u32, bias: ?CF32, m: u32, n: u32, k: u32, flags: u32, alpha: f32) callconv(.nvptx_kernel) void {
    gemmT(f32, out, so_i, so_b, a, sa_i, sa_p, sa_b, w, sw_p, sw_j, sw_b, bias, m, n, k, flags, alpha);
}

inline fn gemmT(comptime W: type, out_base: F32, so_i: u32, so_b: u32, a_base: CF32, sa_i: u32, sa_p: u32, sa_b: u32, w_base: [*]addrspace(.global) const W, sw_p: u32, sw_j: u32, sw_b: u32, bias: ?CF32, m: u32, n: u32, k: u32, flags: u32, alpha: f32) void {
    const t = tid();
    const tx = t % 16;
    const ty = t / 16;
    const bz = ctaZ();
    const out = out_base + bz * so_b;
    const a = a_base + bz * sa_b;
    const w = w_base + bz * sw_b;
    const m0 = ctaY() * BM;
    const n0 = ctaX() * BN;
    const lr = t / 4; // tile row/column this thread loads
    const lk = (t % 4) * 4; // first of 4 k-indices it loads

    var acc: [4]@Vector(4, f32) = @splat(@splat(0));
    var k0: u32 = 0;
    while (k0 < k) : (k0 += BK) {
        const gm = m0 + lr;
        const gn = n0 + lr;
        inline for (0..4) |i| {
            const kk = k0 + lk + @as(u32, i);
            tile_a[lk + i][lr] = if (gm < m and kk < k) a[gm * sa_i + kk * sa_p] else 0;
            tile_w[lk + i][lr] = if (gn < n and kk < k) @floatCast(w[kk * sw_p + gn * sw_j]) else 0;
        }
        barrier();
        inline for (0..BK) |kk| {
            const av = tileA()[kk][ty];
            const wv = tileW()[kk][tx];
            inline for (0..4) |i| acc[i] += @as(@Vector(4, f32), @splat(av[i])) * wv;
        }
        barrier();
    }
    inline for (0..4) |i| {
        const row = m0 + ty * 4 + @as(u32, i);
        if (row < m) {
            inline for (0..4) |j| {
                const col = n0 + tx * 4 + @as(u32, j);
                if (col < n) epilogue(out, row * so_i + col, acc[i][j] * alpha, bias, col, flags);
            }
        }
    }
}

/// y[n] = w[n][k] · x[k] (+ epilogue), for single-token decoding. One warp per
/// output row, 8 rows per 256-thread block; reads 16-byte chunks of f16 weights.
/// Requires k % 8 == 0.
export fn gemv(y: F32, x: CF32, w: CF16, bias: ?CF32, n: u32, k: u32, flags: u32) callconv(.nvptx_kernel) void {
    const t = tid();
    const lane = t % 32;
    const row = ctaX() * 8 + t / 32;
    if (row >= n) return;
    const wr: [*]addrspace(.global) const @Vector(8, f16) = @ptrCast(@alignCast(w + row * k));
    const xr: [*]addrspace(.global) const @Vector(8, f32) = @ptrCast(@alignCast(x));
    var acc: @Vector(8, f32) = @splat(0);
    var c = lane;
    while (c < k / 8) : (c += 32) {
        const wv: @Vector(8, f32) = @floatCast(wr[c]);
        acc += wv * xr[c];
    }
    const s = warpSum(@reduce(.Add, acc));
    if (lane == 0) epilogue(y, row, s, bias, row, flags);
}

// ---------------------------------------------------------------- normalization

/// Row-wise LayerNorm (eps 1e-5). Grid = rows, 256 threads. `out` may alias `x`.
export fn layernorm(out: F32, x: CF32, w: CF32, b: CF32, n: u32) callconv(.nvptx_kernel) void {
    const t = tid();
    const base = ctaX() * n;
    var s: f32 = 0;
    var i = t;
    while (i < n) : (i += 256) s += x[base + i];
    const mean = blockSum(s) / @as(f32, @floatFromInt(n));
    var v: f32 = 0;
    i = t;
    while (i < n) : (i += 256) {
        const d = x[base + i] - mean;
        v += d * d;
    }
    const variance = blockSum(v) / @as(f32, @floatFromInt(n));
    const inv = 1.0 / @sqrt(variance + 1e-5);
    i = t;
    while (i < n) : (i += 256) out[base + i] = (x[base + i] - mean) * inv * w[i] + b[i];
}

// ---------------------------------------------------------------- attention

var att_k: [64][head_dim]f32 align(16) addrspace(.shared) = undefined;
var att_v: [64][head_dim]f32 align(16) addrspace(.shared) = undefined;
inline fn rowsV4(p: *align(16) addrspace(.shared) [64][head_dim]f32) *addrspace(.shared) [64][head_dim / 4]@Vector(4, f32) {
    return @ptrCast(p);
}

/// Full (unmasked) multi-head attention for the encoder, flash-attention style:
/// keys/values stream through shared memory in tiles of 64 and the softmax is
/// computed online, so the n×n score matrix never exists.
/// Grid (ceil(n/64), heads), 64 threads; thread = one query row.
export fn attn_full(out: F32, q: CF32, k: CF32, v: CF32, n: u32, s: u32, scale: f32) callconv(.nvptx_kernel) void {
    @setEvalBranchQuota(10_000);
    const t = tid();
    const h = ctaY();
    const i = ctaX() * 64 + t;
    const valid = i < n;
    const hoff = h * head_dim;

    var qr: [head_dim]f32 = undefined;
    var o: [head_dim]f32 = @splat(0);
    inline for (0..head_dim) |c| qr[c] = if (valid) q[i * s + hoff + c] * (scale * log2e) else 0;
    var mx: f32 = -inf;
    var l: f32 = 0;

    var j0: u32 = 0;
    while (j0 < n) : (j0 += 64) {
        barrier();
        for (0..64) |r| {
            const row = j0 + @as(u32, @intCast(r));
            att_k[r][t] = if (row < n) k[row * s + hoff + t] else 0;
            att_v[r][t] = if (row < n) v[row * s + hoff + t] else 0;
        }
        barrier();
        const cnt = @min(64, n - j0);
        var c16: u32 = 0;
        while (c16 < cnt) : (c16 += 16) {
            var sc: [16]f32 = undefined;
            var cmax = mx;
            inline for (0..16) |jj| {
                const idx = c16 + @as(u32, jj);
                var d: f32 = 0;
                inline for (0..head_dim / 4) |c4| {
                    const kv = rowsV4(&att_k)[idx][c4];
                    const qv: @Vector(4, f32) = qr[c4 * 4 ..][0..4].*;
                    d += @reduce(.Add, kv * qv);
                }
                sc[jj] = if (idx < cnt) d else -inf;
                cmax = @max(cmax, sc[jj]);
            }
            const corr = exp2(mx - cmax);
            mx = cmax;
            l *= corr;
            inline for (0..head_dim) |c| o[c] *= corr;
            inline for (0..16) |jj| {
                const p = exp2(sc[jj] - mx);
                l += p;
                const idx = c16 + @as(u32, jj);
                inline for (0..head_dim / 4) |c4| {
                    const vv = rowsV4(&att_v)[idx][c4];
                    const ov: @Vector(4, f32) = o[c4 * 4 ..][0..4].*;
                    o[c4 * 4 ..][0..4].* = ov + @as(@Vector(4, f32), @splat(p)) * vv;
                }
            }
        }
    }
    if (valid) {
        const inv = 1.0 / l;
        inline for (0..head_dim) |c| out[i * s + hoff + c] = o[c] * inv;
    }
}

const split = shared.split;
const part_stride = shared.part_stride;

var one_q: [head_dim]f32 addrspace(.shared) = undefined;
var one_s: [split]f32 addrspace(.shared) = undefined;
var one_o: [2][head_dim]f32 addrspace(.shared) = undefined;

/// Single-query attention (one decoder token) over n keys, split across blocks
/// so long contexts (1500 audio frames) use the whole GPU.
/// Grid (ceil(n/128), heads), 128 threads. Writes partial results to `part`.
export fn attn_one_part(part: F32, q: CF32, k: CF32, v: CF32, n: u32, s: u32, scale: f32) callconv(.nvptx_kernel) void {
    const t = tid();
    const lane = t % 32;
    const warp = t / 32;
    const h = ctaY();
    const sp = ctaX();
    const nsplit = (n + split - 1) / split;
    const j0 = sp * split;
    const cnt = @min(split, n - j0);
    const hoff = h * head_dim;

    if (t < head_dim) one_q[t] = q[hoff + t] * (scale * log2e);
    barrier();
    // One warp per key: lanes split the 64-dim dot product, so loads coalesce.
    var jj = warp;
    while (jj < cnt) : (jj += 4) {
        const row = (j0 + jj) * s + hoff;
        const d = k[row + lane] * one_q[lane] + k[row + lane + 32] * one_q[lane + 32];
        const tot = warpSum(d);
        if (lane == 0) one_s[jj] = tot;
    }
    barrier();
    const sv = if (t < cnt) one_s[t] else -inf;
    const mx = blockMax(sv);
    const p = if (t < cnt) exp2(sv - mx) else 0;
    one_s[t] = p;
    const l = blockSum(p); // includes a barrier, so one_s is complete after this
    const g = t / head_dim;
    const c = t % head_dim;
    var acc: f32 = 0;
    jj = g;
    while (jj < cnt) : (jj += 2) acc += one_s[jj] * v[(j0 + jj) * s + hoff + c];
    one_o[g][c] = acc;
    barrier();
    const base = (h * nsplit + sp) * part_stride;
    if (t < head_dim) part[base + 2 + t] = one_o[0][t] + one_o[1][t];
    if (t == 0) {
        part[base] = mx;
        part[base + 1] = l;
    }
}

/// Merges the per-split partial softmaxes. Grid = heads, 64 threads.
export fn attn_one_combine(out: F32, part: CF32, nsplit: u32) callconv(.nvptx_kernel) void {
    const h = ctaX();
    const c = tid();
    var mx: f32 = -inf;
    for (0..nsplit) |sp| mx = @max(mx, part[(h * nsplit + @as(u32, @intCast(sp))) * part_stride]);
    var l: f32 = 0;
    var o: f32 = 0;
    for (0..nsplit) |sp| {
        const base = (h * nsplit + @as(u32, @intCast(sp))) * part_stride;
        const f = exp2(part[base] - mx);
        l += part[base + 1] * f;
        o += part[base + 2 + c] * f;
    }
    out[h * head_dim + c] = o / l;
}

// ---------------------------------------------------------------- data movement

/// conv1 im2col (kernel 3, pad 1, stride 1): col[t][c*3+kk] = spec[c][t+kk-1].
export fn im2col1(col: F32, spec: CF32, frames: u32, mels: u32) callconv(.nvptx_kernel) void {
    const idx = ctaX() * 256 + tid();
    const per = mels * 3;
    if (idx >= frames * per) return;
    const t = idx / per;
    const r = idx % per;
    const c = r / 3;
    const src = @as(i32, @intCast(t + r % 3)) - 1;
    col[idx] = if (src < 0 or src >= frames) 0 else spec[c * frames + @as(u32, @intCast(src))];
}

/// conv2 im2col (kernel 3, pad 1, stride 2): col[t][c*3+kk] = h1[2t+kk-1][c].
export fn im2col2(col: F32, h1: CF32, ctx: u32, frames: u32, state: u32) callconv(.nvptx_kernel) void {
    const idx = ctaX() * 256 + tid();
    const per = state * 3;
    if (idx >= ctx * per) return;
    const t = idx / per;
    const r = idx % per;
    const c = r / 3;
    const src = @as(i32, @intCast(2 * t + r % 3)) - 1;
    col[idx] = if (src < 0 or src >= frames) 0 else h1[@as(u32, @intCast(src)) * state + c];
}

/// x = tok_emb[token] + pos_emb[pos]. Grid ceil(s/256), 256 threads.
export fn embed(x: F32, tok_emb: CF16, pos_emb: CF32, token: u32, pos: u32, s: u32) callconv(.nvptx_kernel) void {
    const c = ctaX() * 256 + tid();
    if (c < s) x[c] = @as(f32, @floatCast(tok_emb[token * s + c])) + pos_emb[pos * s + c];
}

var pick_v: [32]f32 addrspace(.shared) = undefined;
var pick_i: [32]u32 addrspace(.shared) = undefined;

/// Greedy token choice on the GPU, so only 12 bytes cross PCIe per token
/// instead of the whole logit vector. Tokens >= ban_from and the two ids
/// ban_a/ban_b are excluded. Writes res = { argmax, logprob(argmax),
/// logprob(nosp) } (log-probs over the allowed tokens). One block, 1024 threads.
export fn pick(res: U32, logits: CF32, n: u32, ban_from: u32, ban_a: u32, ban_b: u32, nosp: u32) callconv(.nvptx_kernel) void {
    const t = tid();
    var bv: f32 = -inf;
    var bi: u32 = 0;
    var i = t;
    while (i < n) : (i += 1024) {
        if (i >= ban_from or i == ban_a or i == ban_b) continue;
        const v = logits[i];
        if (v > bv) {
            bv = v;
            bi = i;
        }
    }
    // Arg-max reduction; ties go to the lower index (matches the CPU argmax).
    inline for (.{ 16, 8, 4, 2, 1 }) |o| {
        const ov = shflXor(bv, o);
        const oi = shflXorU(bi, o);
        if (ov > bv or (ov == bv and oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (t % 32 == 0) {
        pick_v[t / 32] = bv;
        pick_i[t / 32] = bi;
    }
    barrier();
    bv = pick_v[0];
    bi = pick_i[0];
    for (1..32) |wi| {
        if (pick_v[wi] > bv or (pick_v[wi] == bv and pick_i[wi] < bi)) {
            bv = pick_v[wi];
            bi = pick_i[wi];
        }
    }
    var s: f32 = 0;
    i = t;
    while (i < n) : (i += 1024) {
        if (i >= ban_from or i == ban_a or i == ban_b) continue;
        s += exp2((logits[i] - bv) * log2e);
    }
    const sum = blockSum(s);
    if (t == 0) {
        const lse = bv + log2(sum) * ln2;
        const banned = nosp >= ban_from or nosp == ban_a or nosp == ban_b;
        res[0] = bi;
        res[1] = @bitCast(bv - lse);
        res[2] = @bitCast(if (banned) -inf else logits[nosp] - lse);
    }
}

// ---------------------------------------------------------------- log-mel front end

const n_fft = shared.n_fft;
const n_bins = shared.n_bins;

/// Zero-pads audio to `total` samples, then reflect-pads n_fft/2 on each side
/// (torch.stft center=True). Grid ceil((total+n_fft)/256), 256 threads.
export fn pad_audio(padded: F32, samples: CF32, n: u32, total: u32) callconv(.nvptx_kernel) void {
    const idx = ctaX() * 256 + tid();
    const pad = n_fft / 2;
    if (idx >= total + 2 * pad) return;
    // Index into the 30 s signal, with reflection at both ends.
    const j: u32 = if (idx < pad) pad - idx else if (idx >= pad + total) 2 * total + pad - 2 - idx else idx - pad;
    padded[idx] = if (j < n) samples[j] else 0;
}

var mel_pw: [n_bins]f32 addrspace(.shared) = undefined;

/// From the STFT (spec[f][0..201] = Re, spec[f][201..402] = Im), computes
/// log10 mel energies: out[m][f]. Grid = frames, 128 threads.
export fn mel_power(out: F32, spec: CF32, filters: CF32, n_mels: u32, frames: u32) callconv(.nvptx_kernel) void {
    const f = ctaX();
    const t = tid();
    var k = t;
    while (k < n_bins) : (k += 128) {
        const re = spec[f * 2 * n_bins + k];
        const im = spec[f * 2 * n_bins + n_bins + k];
        mel_pw[k] = re * re + im * im;
    }
    barrier();
    var m = t;
    while (m < n_mels) : (m += 128) {
        var s: f32 = 0;
        for (0..n_bins) |b| s += filters[m * n_bins + @as(u32, @intCast(b))] * mel_pw[b];
        out[m * frames + f] = log2(@max(s, 1e-10)) * 0.30102999566398120; // log10
    }
}

/// res[0] = max(x). One block, 1024 threads.
export fn reduce_max(res: F32, x: CF32, n: u32) callconv(.nvptx_kernel) void {
    var mx: f32 = -inf;
    var i = tid();
    while (i < n) : (i += 1024) mx = @max(mx, x[i]);
    const r = blockMax(mx);
    if (tid() == 0) res[0] = r;
}

/// x = (max(x, max_all - 8) + 4) / 4. Grid ceil(n/256), 256 threads.
export fn mel_norm(x: F32, n: u32, mx: CF32) callconv(.nvptx_kernel) void {
    const i = ctaX() * 256 + tid();
    if (i < n) x[i] = (@max(x[i], mx[0] - 8.0) + 4.0) / 4.0;
}

// ---------------------------------------------------------------- training
//
// Teacher-forced decoder forward/backward with LoRA adapters, and the
// optimizers. Matrix products go through `gemm`; these are the rest.

inline fn atomicAdd(p: *addrspace(.global) f32, v: f32) void {
    _ = @atomicRmw(f32, p, .Add, v, .monotonic);
}

/// x[l] = tok_emb[tokens[l]] + pos_emb[l]. Grid (ceil(s/256), L), 256 threads.
export fn embed_seq(x: F32, tok_emb: CF16, pos_emb: CF32, tokens: [*]addrspace(.global) const u32, s: u32) callconv(.nvptx_kernel) void {
    const c = ctaX() * 256 + tid();
    const l = ctaY();
    if (c < s) x[l * s + c] = @as(f32, @floatCast(tok_emb[tokens[l] * s + c])) + pos_emb[l * s + c];
}

/// LayerNorm that also saves each row's mean and 1/std for the backward pass.
/// Grid = rows, 256 threads.
export fn layernorm_fwd(out: F32, x: CF32, w: CF32, b: CF32, n: u32, mean_out: F32, rstd_out: F32) callconv(.nvptx_kernel) void {
    const t = tid();
    const row = ctaX();
    const base = row * n;
    var s: f32 = 0;
    var i = t;
    while (i < n) : (i += 256) s += x[base + i];
    const mean = blockSum(s) / @as(f32, @floatFromInt(n));
    var v: f32 = 0;
    i = t;
    while (i < n) : (i += 256) {
        const d = x[base + i] - mean;
        v += d * d;
    }
    const rstd = 1.0 / @sqrt(blockSum(v) / @as(f32, @floatFromInt(n)) + 1e-5);
    if (t == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }
    i = t;
    while (i < n) : (i += 256) out[base + i] = (x[base + i] - mean) * rstd * w[i] + b[i];
}

/// dx += LayerNorm backward (gamma/beta frozen). Grid = rows, 256 threads.
export fn layernorm_back(dx: F32, dy: CF32, x: CF32, w: CF32, mean: CF32, rstd: CF32, n: u32) callconv(.nvptx_kernel) void {
    const t = tid();
    const row = ctaX();
    const base = row * n;
    const mu = mean[row];
    const rs = rstd[row];
    var s1: f32 = 0;
    var s2: f32 = 0;
    var i = t;
    while (i < n) : (i += 256) {
        const d = dy[base + i] * w[i];
        s1 += d;
        s2 += d * (x[base + i] - mu) * rs;
    }
    const nf: f32 = @floatFromInt(n);
    const m1 = blockSum(s1) / nf;
    const m2 = blockSum(s2) / nf;
    i = t;
    while (i < n) : (i += 256) {
        const xh = (x[base + i] - mu) * rs;
        dx[base + i] += rs * (dy[base + i] * w[i] - m1 - xh * m2);
    }
}

/// In-place row softmax of attention scores [rows][cols], where row r is
/// query r % lq. With `causal`, query i only sees keys 0..i (the rest become 0).
/// Grid = rows, 256 threads.
export fn softmax_rows(p: F32, cols: u32, lq: u32, causal: u32) callconv(.nvptx_kernel) void {
    const t = tid();
    const row = ctaX();
    const base = row * cols;
    const n = if (causal != 0) @min(cols, row % lq + 1) else cols;
    var mx: f32 = -inf;
    var j = t;
    while (j < n) : (j += 256) mx = @max(mx, p[base + j]);
    mx = blockMax(mx);
    var s: f32 = 0;
    j = t;
    while (j < n) : (j += 256) {
        const e = exp2((p[base + j] - mx) * log2e);
        p[base + j] = e;
        s += e;
    }
    const inv = 1.0 / blockSum(s);
    j = t;
    while (j < cols) : (j += 256) p[base + j] = if (j < n) p[base + j] * inv else 0;
}

/// Softmax backward, in place: ds = p ⊙ (dp − Σ p·dp) · scale (dp is overwritten).
/// Grid = rows, 256 threads.
export fn softmax_back(ds: F32, p: CF32, cols: u32, scale: f32) callconv(.nvptx_kernel) void {
    const t = tid();
    const base = ctaX() * cols;
    var d: f32 = 0;
    var j = t;
    while (j < cols) : (j += 256) d += p[base + j] * ds[base + j];
    const dot = blockSum(d);
    j = t;
    while (j < cols) : (j += 256) ds[base + j] = p[base + j] * (ds[base + j] - dot) * scale;
}

/// g = gelu(u). Grid ceil(n/256), 256 threads.
export fn gelu_fwd(g: F32, u: CF32, n: u32) callconv(.nvptx_kernel) void {
    const i = ctaX() * 256 + tid();
    if (i < n) g[i] = gelu(u[i]);
}

/// dg *= gelu'(u). Grid ceil(n/256), 256 threads.
export fn gelu_back(dg: F32, u: CF32, n: u32) callconv(.nvptx_kernel) void {
    const i = ctaX() * 256 + tid();
    if (i >= n) return;
    const x = u[i];
    const c: f32 = 0.7978845608028654;
    const y = c * (x + 0.044715 * x * x * x);
    const th = 1.0 - 2.0 / (exp2(2.0 * log2e * y) + 1.0);
    dg[i] *= 0.5 * (1 + th) + 0.5 * x * (1 - th * th) * c * (1 + 3 * 0.044715 * x * x);
}

/// Cross-entropy over logits [rows][V]: adds mean loss · inv_n to loss[0]
/// and overwrites the logits with d(loss)/d(logits) = (softmax − onehot)·inv_n.
/// Grid = rows, 1024 threads.
export fn xent(logits: F32, targets: [*]addrspace(.global) const u32, v: u32, loss: F32, inv_n: f32) callconv(.nvptx_kernel) void {
    const t = tid();
    const row = ctaX();
    const base = row * v;
    var mx: f32 = -inf;
    var j = t;
    while (j < v) : (j += 1024) mx = @max(mx, logits[base + j]);
    mx = blockMax(mx);
    var s: f32 = 0;
    j = t;
    while (j < v) : (j += 1024) s += exp2((logits[base + j] - mx) * log2e);
    const sum = blockSum(s);
    const tgt = targets[row];
    const lse = mx + log2(sum) * ln2;
    if (t == 0) atomicAdd(&loss[0], (lse - logits[base + tgt]) * inv_n);
    barrier(); // everyone has read logits[base + tgt] before it is overwritten
    const inv = 1.0 / sum;
    j = t;
    while (j < v) : (j += 1024) {
        const pj = exp2((logits[base + j] - mx) * log2e) * inv;
        logits[base + j] = (pj - @as(f32, if (j == tgt) 1 else 0)) * inv_n;
    }
}

/// acc[0] += Σ x². Grid ceil(n/1024) (capped), 256 threads.
export fn sumsq(x: CF32, n: u32, acc: F32) callconv(.nvptx_kernel) void {
    var s: f32 = 0;
    var i = ctaX() * 256 + tid();
    const stride = 256 * nctaX();
    while (i < n) : (i += stride) s += x[i] * x[i];
    const tot = blockSum(s);
    if (tid() == 0) atomicAdd(&acc[0], tot);
}

/// Averages a summed batch gradient and clips its norm:
/// g *= inv_batch · min(1, max_norm / ‖g·inv_batch‖), with ‖g‖² = sq[0].
export fn clip_scale(g: F32, n: u32, sq: CF32, inv_batch: f32, max_norm: f32) callconv(.nvptx_kernel) void {
    const i = ctaX() * 256 + tid();
    if (i >= n) return;
    const norm = @sqrt(sq[0]) * inv_batch;
    const k = if (norm > max_norm) inv_batch * max_norm / norm else inv_batch;
    g[i] *= k;
}

/// Adam step. c1/c2 are the bias corrections 1-β₁ᵗ and 1-β₂ᵗ.
export fn adam(p: F32, g: CF32, m: F32, v: F32, n: u32, lr: f32, c1: f32, c2: f32) callconv(.nvptx_kernel) void {
    const i = ctaX() * 256 + tid();
    if (i >= n) return;
    const b1: f32 = 0.9;
    const b2: f32 = 0.999;
    m[i] = b1 * m[i] + (1 - b1) * g[i];
    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    p[i] -= lr * (m[i] / c1) / (@sqrt(v[i] / c2) + 1e-8);
}

pub const max_short_side = 32; // LoRA rank limit for the Muon kernel

var ns_gram: [max_short_side][max_short_side]f32 addrspace(.shared) = undefined;
var ns_poly: [max_short_side][max_short_side]f32 addrspace(.shared) = undefined;

/// Muon: Nesterov momentum, then orthogonalize the update with 5 quintic
/// Newton–Schulz steps, then w -= lr·0.2·sqrt(max(rows,cols))·update.
/// One block (256 threads) per parameter matrix; desc[3b..] = { offset, rows,
/// cols } in floats. `x` is scratch the size of the parameters. The short
/// side must be ≤ 32 (it is the LoRA rank).
export fn muon(p: F32, g: CF32, mom: F32, x_all: F32, desc: [*]addrspace(.global) const u32, beta: f32, lr: f32) callconv(.nvptx_kernel) void {
    const t = tid();
    const off = desc[ctaX() * 3];
    const rows = desc[ctaX() * 3 + 1];
    const cols = desc[ctaX() * 3 + 2];
    const tall = rows > cols;
    const ms = if (tall) cols else rows; // short side
    const nl = if (tall) rows else cols; // long side
    const x = x_all + off;
    const total = rows * cols;

    // u = g + β·(mom = β·mom + g), stored short-side-major in x.
    var ss: f32 = 0;
    var idx = t;
    while (idx < total) : (idx += 256) {
        const gv = g[off + idx];
        const mv = beta * mom[off + idx] + gv;
        mom[off + idx] = mv;
        const u = gv + beta * mv;
        const r = idx / cols;
        const c = idx % cols;
        x[if (tall) c * nl + r else idx] = u;
        ss += u * u;
    }
    const norm = @sqrt(blockSum(ss)); // blockSum's barrier also publishes x
    if (norm < 1e-12) return;
    const inv = 1.0 / (norm + 1e-7);
    idx = t;
    while (idx < total) : (idx += 256) x[idx] *= inv;
    barrier();

    const ca: f32 = 3.4445;
    const cb: f32 = -4.7750;
    const cc: f32 = 2.0315;
    for (0..5) |_| {
        // gram = X·Xᵀ
        var pr = t;
        while (pr < ms * ms) : (pr += 256) {
            const i = pr / ms;
            const j = pr % ms;
            var d: f32 = 0;
            for (0..nl) |q| d += x[i * nl + @as(u32, @intCast(q))] * x[j * nl + @as(u32, @intCast(q))];
            ns_gram[i][j] = d;
        }
        barrier();
        // poly = b·gram + c·gram²
        pr = t;
        while (pr < ms * ms) : (pr += 256) {
            const i = pr / ms;
            const j = pr % ms;
            var d: f32 = 0;
            for (0..ms) |q| d += ns_gram[i][q] * ns_gram[q][j];
            ns_poly[i][j] = cb * ns_gram[i][j] + cc * d;
        }
        barrier();
        // X = a·X + poly·X, one column per thread.
        var col = t;
        while (col < nl) : (col += 256) {
            var xc: [max_short_side]f32 = undefined;
            for (0..ms) |q| xc[q] = x[@as(u32, @intCast(q)) * nl + col];
            for (0..ms) |i| {
                var d: f32 = ca * xc[i];
                for (0..ms) |q| d += ns_poly[i][q] * xc[q];
                x[@as(u32, @intCast(i)) * nl + col] = d;
            }
        }
        barrier();
    }
    const kstep = lr * 0.2 * @sqrt(@as(f32, @floatFromInt(nl)));
    idx = t;
    while (idx < total) : (idx += 256) {
        const r = idx / cols;
        const c = idx % cols;
        p[off + idx] -= kstep * x[if (tall) c * nl + r else idx];
    }
}

/// Folds a LoRA adapter into f16 weights: w[o][i] += scale · Σ_r b[o][r]·a[r][i].
/// Grid ceil(n_out·n_in/256), 256 threads.
export fn lora_merge(w: [*]addrspace(.global) f16, a: CF32, b: CF32, n_out: u32, n_in: u32, rank: u32, scale: f32) callconv(.nvptx_kernel) void {
    const idx = ctaX() * 256 + tid();
    if (idx >= n_out * n_in) return;
    const o = idx / n_in;
    const i = idx % n_in;
    var d: f32 = 0;
    for (0..rank) |r| {
        const rr: u32 = @intCast(r);
        d += b[o * rank + rr] * a[rr * n_in + i];
    }
    w[idx] = @floatCast(@as(f32, @floatCast(w[idx])) + scale * d);
}
