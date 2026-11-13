//! The two QNN graphs of the speech transformer, built op by op from QNN
//! built-ins (no custom ops: Windows only loads signed Hexagon code).
//! The math follows gpu/engine.zig exactly:
//!
//! Encoder (static shapes, one 30 s window):
//!   audio [3002][160]   reflect-padded samples, scaled by 2^k on the host
//!                       (see `Engine.encode`) so FP16 keeps the quiet bins
//!   mel_bias [1]        -log10(2^k · 2^k · filter_scale) / 4
//!   -> frames (3 slices + concat) -> windowed DFT (2 FullyConnected)
//!   -> power -> mel FullyConnected -> ln -> clamp to max-8 -> (x+4)/4
//!   -> conv1/conv2 (Conv2d 1x3) + GELU -> + positions -> blocks -> ln_post
//!   -> per decoder layer: cross K, V as [heads][1500][64]
//!
//! Decoder step (one token, KV cache of n_text_ctx positions passed in):
//!   token, pos [1] i32  embedded in the graph (Gather on the token table and
//!                       the position table, then Add)
//!   mask [1][1][ctx+1]  0 for cached positions < pos, -30000 otherwise; the
//!                       last entry is the token itself (always 0)
//!   sk/sv per layer     [heads][ctx][64]   self-attention cache
//!   ck/cv per layer     [heads][1500][64]  from the encoder
//!   ban [1][n_vocab]    0 = allowed, -30000 = banned token
//!   nosp [1] i32        the no-speech token id
//!   -> pick [1][1] i32 (Argmax of logits + ban), lp / nlp [1][1] (LogSoftmax
//!      of logits + ban at the pick and at nosp), and kn/vn per layer
//!      [1][state] (this position's K/V, which the host copies into the cache)
//!
//! The attention scale 1/sqrt(64) = 1/8 is folded into the query weights
//! (exact in FP16).
const std = @import("std");
const qnn = @import("qnn");
const graph = @import("graph.zig");
const model_mod = @import("../model.zig");
const mel = @import("../mel.zig");
const adapter = @import("../adapter.zig");
const Model = model_mod.Model;
const Linear = model_mod.Linear;
const Norm = model_mod.Norm;
const Builder = graph.Builder;
const T = graph.T;
const ops = qnn.ops;

pub const head_dim = 64;
pub const n_bins = mel.n_fft / 2 + 1; // 201
/// Rows of 160 samples fed to the encoder: frame t uses rows t, t+1 and the
/// first 80 samples of row t+2.
pub const audio_rows = mel.n_frames + 2;
pub const mask_off: f32 = -30000.0;

pub const Gelu = enum {
    /// 0.5·u·(1 + tanh(√(2/π)(u + 0.044715u³))) = u·sigmoid(2·...), what
    /// the GPU computes. Five element-wise ops + Sigmoid.
    tanh,
    /// QNN's Gelu op (erf form, what the model was trained with).
    builtin,
};

pub const Config = struct {
    gelu: Gelu = .tanh,
    /// Floor applied to the mel energies before the log (scaled units).
    mel_floor: f32 = 0x1p-24,
};

/// Power of two the mel filterbank is multiplied by, so the largest
/// possible mel energy stays <= the largest power value (FP16 range).
pub fn filterScaleLog2(m: *const Model) i32 {
    var best: f32 = 0;
    for (0..m.hp.n_mels) |r| {
        var s: f32 = 0;
        for (m.mel_filters[r * n_bins ..][0..n_bins]) |v| s += @abs(v);
        best = @max(best, s);
    }
    if (best <= 0) return 0;
    return @intFromFloat(@floor(-std.math.log2(best)));
}

/// Weights of the decoder's adapted layers: base or with a LoRA folded in.
pub const DecWeights = struct {
    /// [layer][slot] -> f16 [n_out][n_in]
    w: []const [adapter.n_slots][]const f16,
};

fn linear(b: *Builder, x: T, l: Linear, w: []const f16, scale: f32) !T {
    const wt = if (scale == 1)
        try b.constF16(&.{ @intCast(l.n_out), @intCast(l.n_in) }, w)
    else blk: {
        const s = try b.arena.alloc(f16, w.len);
        for (s, w) |*o, v| o.* = @floatCast(@as(f32, @floatCast(v)) * scale);
        break :blk try b.constF16(&.{ @intCast(l.n_out), @intCast(l.n_in) }, s);
    };
    const bias = if (l.b) |bv| blk: {
        if (scale == 1) break :blk try b.constF32(&.{@intCast(l.n_out)}, bv);
        const s = try b.arena.alloc(f32, bv.len);
        for (s, bv) |*o, v| o.* = v * scale;
        break :blk try b.constF32(&.{@intCast(l.n_out)}, s);
    } else null;
    return b.fc(x, wt, bias);
}

fn norm(b: *Builder, x: T, n: Norm) !T {
    const d: u32 = @intCast(n.w.len);
    return b.layerNorm(x, try b.constF32(&.{d}, n.w), try b.constF32(&.{d}, n.b));
}

fn gelu(b: *Builder, cfg: Config, u: T) !T {
    switch (cfg.gelu) {
        .builtin => return b.unary(ops.gelu.name, u),
        .tanh => {
            const k0: f32 = 2.0 * 0.7978845608028654;
            const k1: f32 = k0 * 0.044715;
            const sq = try b.mul(u, u);
            const t = try b.add(try b.mul(sq,try b.scalar(k1)), try b.scalar(k0));
            const s = try b.unary(ops.sigmoid.name, try b.mul(u, t));
            return b.mul(u, s);
        },
    }
}

/// [n][heads*64] -> [heads][n][64]
fn splitHeads(b: *Builder, x: T, heads: u32) !T {
    const n = x.dims[0];
    return b.transpose(try b.reshape(x, &.{ n, heads, head_dim }), &.{ 1, 0, 2 });
}

/// [heads][n][64] -> [n][heads*64]
fn mergeHeads(b: *Builder, x: T) !T {
    const heads = x.dims[0];
    const n = x.dims[1];
    return b.reshape(try b.transpose(x, &.{ 1, 0, 2 }), &.{ n, heads * head_dim });
}

/// Conv1d weights [out][in][3] as a QNN Conv2d filter [1][3][in][out].
fn convFilter(b: *Builder, l: Linear) !T {
    const n_out = l.n_out;
    const n_in = l.n_in / 3;
    const f = try b.arena.alloc(f16, l.w.len);
    for (0..n_out) |o| for (0..n_in) |i| for (0..3) |k| {
        f[(k * n_in + i) * n_out + o] = l.w[o * l.n_in + i * 3 + k];
    };
    return b.constF16(&.{ 1, 3, @intCast(n_in), @intCast(n_out) }, f);
}

/// Log-mel spectrogram [3000][n_mels] (already (x+4)/4-normalized).
pub fn frontEnd(b: *Builder, m: *const Model, cfg: Config, audio: T, mel_bias: T) !T {
    const T_: u32 = mel.n_frames;
    const hop: u32 = mel.hop;
    const r0 = try b.slice(audio, &.{ 0, 0 }, &.{ T_, hop });
    const r1 = try b.slice(audio, &.{ 1, 0 }, &.{ T_ + 1, hop });
    const r2 = try b.slice(audio, &.{ 2, 0 }, &.{ T_ + 2, mel.n_fft - 2 * hop });
    const frames = try b.concat(&.{ r0, r1, r2 }, 1); // [3000][400]
    const basis = mel.dftBasis();
    const cos_w = try b.constF32(&.{ n_bins, mel.n_fft }, @as([*]const f32, @ptrCast(&basis.cos))[0 .. n_bins * mel.n_fft]);
    const sin_w = try b.constF32(&.{ n_bins, mel.n_fft }, @as([*]const f32, @ptrCast(&basis.sin))[0 .. n_bins * mel.n_fft]);
    const re = try b.fc(frames, cos_w, null);
    const im = try b.fc(frames, sin_w, null);
    const power = try b.add(try b.mul(re, re), try b.mul(im, im)); // [3000][201]

    const n_mels: u32 = @intCast(m.hp.n_mels);
    const g = std.math.ldexp(@as(f32, 1), filterScaleLog2(m));
    const filt = try b.arena.alloc(f32, m.mel_filters.len);
    for (filt, m.mel_filters) |*o, v| o.* = v * g;
    const melw = try b.constF32(&.{ n_mels, n_bins }, filt);
    var x = try b.fc(power, melw, null); // [3000][n_mels]
    x = try b.max(x, try b.scalar(cfg.mel_floor));
    x = try b.unary(ops.element_wise_log.name, x);
    // y = log10(mel)/4 = ln(mel') · (1/(4 ln 10)) + mel_bias
    x = try b.add(try b.mul(x, try b.scalar(0.25 / std.math.ln10)), mel_bias);
    x = try b.max(x, try b.scalar(-10.0 / 4.0)); // log10 floor 1e-10
    const mx = try b.reduceMaxAll(x);
    x = try b.max(x, try b.add(mx, try b.scalar(-8.0 / 4.0)));
    return b.add(x, try b.scalar(1.0));
}

pub const EncoderStop = enum { full, mel };

/// Builds the encoder graph. Inputs: audio, mel_bias. Outputs: ck0, cv0,
/// ck1, cv1, ... (or just `mel` with stop = .mel, for testing).
pub fn encoder(b: *Builder, m: *const Model, cfg: Config, stop: EncoderStop) !void {
    const hp = m.hp;
    const A: u32 = @intCast(hp.n_audio_ctx);
    const ns: u32 = @intCast(hp.n_audio_state);
    const heads: u32 = @intCast(hp.n_audio_head);
    const n_mels: u32 = @intCast(hp.n_mels);
    const audio = try b.input("audio", &.{ audio_rows, mel.hop });
    const mel_bias = try b.input("mel_bias", &.{1});
    const spec = try frontEnd(b, m, cfg, audio, mel_bias);
    if (stop == .mel) {
        _ = try b.output("mel", spec, &.{ mel.n_frames, n_mels });
        return;
    }

    var x = try b.reshape(spec, &.{ 1, 1, mel.n_frames, n_mels });
    const b1 = try b.constF32(&.{ns}, m.conv1.b.?);
    x = try gelu(b, cfg, try b.conv2d(x, try convFilter(b, m.conv1), b1, .{ 1, 1 }, .{ .{ 0, 0 }, .{ 1, 1 } }));
    const b2 = try b.constF32(&.{ns}, m.conv2.b.?);
    x = try gelu(b, cfg, try b.conv2d(x, try convFilter(b, m.conv2), b2, .{ 1, 2 }, .{ .{ 0, 0 }, .{ 1, 1 } }));
    x = try b.reshape(x, &.{ A, ns });
    x = try b.add(x, try b.constF32(&.{ A, ns }, m.enc_pos));

    const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
    for (m.enc_layers) |l| {
        var h = try norm(b, x, l.attn_ln);
        const q = try splitHeads(b, try linear(b, h, l.attn.q, l.attn.q.w, scale), heads);
        const k = try splitHeads(b, try linear(b, h, l.attn.k, l.attn.k.w, 1), heads);
        const v = try splitHeads(b, try linear(b, h, l.attn.v, l.attn.v.w, 1), heads);
        const p = try b.softmax(try b.matmul(q, k, false, true)); // [H][A][A]
        const o = try mergeHeads(b, try b.matmul(p, v, false, false));
        x = try b.add(x, try linear(b, o, l.attn.o, l.attn.o.w, 1));
        h = try norm(b, x, l.mlp_ln);
        const f = try gelu(b, cfg, try linear(b, h, l.fc1, l.fc1.w, 1));
        x = try b.add(x, try linear(b, f, l.fc2, l.fc2.w, 1));
    }
    x = try norm(b, x, m.enc_ln_post);

    const th: u32 = @intCast(hp.n_text_head);
    var name_buf: [16]u8 = undefined;
    for (m.dec_layers, 0..) |l, i| {
        const ck = try splitHeads(b, try linear(b, x, l.cross.k, l.cross.k.w, 1), th);
        _ = try b.output(try dupName(b, &name_buf, "ck", i), ck, ck.dims);
        const cv = try splitHeads(b, try linear(b, x, l.cross.v, l.cross.v.w, 1), th);
        _ = try b.output(try dupName(b, &name_buf, "cv", i), cv, cv.dims);
    }
}

fn dupName(b: *Builder, buf: []u8, prefix: []const u8, i: usize) ![:0]const u8 {
    const s = try std.fmt.bufPrint(buf, "{s}{d}", .{ prefix, i });
    return b.arena.dupeSentinel(u8, s, 0);
}

/// Builds the decoder step graph. Inputs: x, mask, then per layer sk, sv,
/// ck, cv. Outputs: per layer kn, vn, then logits.
pub fn decoder(b: *Builder, m: *const Model, cfg: Config, dw: DecWeights) !void {
    const hp = m.hp;
    const A: u32 = @intCast(hp.n_audio_ctx);
    const ts: u32 = @intCast(hp.n_text_state);
    const heads: u32 = @intCast(hp.n_text_head);
    const ctx: u32 = @intCast(hp.n_text_ctx);
    const i32_t = qnn.c.QNN_DATATYPE_INT_32;
    const token = try b.inputTyped("token", i32_t, &.{1});
    const pos = try b.inputTyped("pos", i32_t, &.{1});
    const mask = try b.input("mask", &.{ 1, 1, ctx + 1 });
    const ban = try b.input("ban", &.{ 1, @intCast(hp.n_vocab) });
    const nosp = try b.inputTyped("nosp", i32_t, &.{1});
    // Token + position embedding (the token table is also the output projection).
    const emb = try b.constF16(&.{ @intCast(hp.n_vocab), ts }, m.tok_emb);
    var x = try b.add(try b.gather(emb, token), try b.gather(try b.constF32(&.{ ctx, ts }, m.dec_pos), pos));
    const L = hp.n_text_layer;
    var name_buf: [16]u8 = undefined;
    const sk = try b.arena.alloc(T, L);
    const sv = try b.arena.alloc(T, L);
    const ck = try b.arena.alloc(T, L);
    const cv = try b.arena.alloc(T, L);
    for (0..L) |i| {
        sk[i] = try b.input(try dupName(b, &name_buf, "sk", i), &.{ heads, ctx, head_dim });
        sv[i] = try b.input(try dupName(b, &name_buf, "sv", i), &.{ heads, ctx, head_dim });
        ck[i] = try b.input(try dupName(b, &name_buf, "ck", i), &.{ heads, A, head_dim });
        cv[i] = try b.input(try dupName(b, &name_buf, "cv", i), &.{ heads, A, head_dim });
    }

    const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
    const one: []const u32 = &.{ heads, 1, head_dim };
    for (m.dec_layers, 0..) |l, i| {
        const w = dw.w[i];
        // self-attention over the cache plus this token
        var h = try norm(b, x, l.attn_ln);
        const q = try linear(b, h, l.attn.q, w[@intFromEnum(adapter.Slot.q)], scale);
        const k = try linear(b, h, l.attn.k, w[@intFromEnum(adapter.Slot.k)], 1);
        const v = try linear(b, h, l.attn.v, w[@intFromEnum(adapter.Slot.v)], 1);
        _ = try b.output(try dupName(b, &name_buf, "kn", i), k, &.{ 1, ts });
        _ = try b.output(try dupName(b, &name_buf, "vn", i), v, &.{ 1, ts });
        const kk = try b.concat(&.{ sk[i], try b.reshape(k, one) }, 1);
        const vv = try b.concat(&.{ sv[i], try b.reshape(v, one) }, 1);
        const s = try b.add(try b.matmul(try b.reshape(q, one), kk, false, true), mask);
        var o = try b.matmul(try b.softmax(s), vv, false, false);
        x = try b.add(x, try linear(b, try b.reshape(o, &.{ 1, ts }), l.attn.o, w[@intFromEnum(adapter.Slot.o)], 1));

        // cross-attention over the audio
        h = try norm(b, x, l.cross_ln);
        const cq = try linear(b, h, l.cross.q, w[@intFromEnum(adapter.Slot.cross_q)], scale);
        const cs = try b.matmul(try b.reshape(cq, one), ck[i], false, true);
        o = try b.matmul(try b.softmax(cs), cv[i], false, false);
        x = try b.add(x, try linear(b, try b.reshape(o, &.{ 1, ts }), l.cross.o, w[@intFromEnum(adapter.Slot.cross_o)], 1));

        h = try norm(b, x, l.mlp_ln);
        const f = try gelu(b, cfg, try linear(b, h, l.fc1, w[@intFromEnum(adapter.Slot.fc1)], 1));
        x = try b.add(x, try linear(b, f, l.fc2, w[@intFromEnum(adapter.Slot.fc2)], 1));
    }
    x = try norm(b, x, m.dec_ln);
    const logits = try b.fc(x, emb, null);

    // Greedy pick, as the GPU's pick kernel: banned ids get `ban` = -30000,
    // so they never win and add nothing to the softmax sum.
    const V: u32 = @intCast(hp.n_vocab);
    const z = try b.add(logits, ban);
    _ = try b.nodeOut(ops.argmax.name, &.{
        qnn.scalarParam(ops.argmax.param.axis, u32, 1),
        qnn.scalarParam(ops.argmax.param.keep_dims, bool, true),
    }, &.{z}, "pick", i32_t, &.{ 1, 1 });
    const ls = try b.node(ops.log_softmax.name, &.{
        qnn.scalarParam(ops.log_softmax.param.axis, u32, 1),
        qnn.scalarParam(ops.log_softmax.param.beta, f32, 1.0),
    }, &.{z}, &.{ 1, V });
    // log-prob of the pick = max of the log-softmax
    _ = try b.output("lp", try b.reduceMaxAll(ls), &.{ 1, 1 });
    _ = try b.output("nlp", try b.gather(try b.reshape(ls, &.{ V, 1 }), nosp), &.{ 1, 1 });
}
