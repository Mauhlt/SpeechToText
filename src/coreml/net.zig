//! The speech transformer as two Core ML ML Programs, built with
//! TranslateCoreML's MIL builder from the ggml weights (fp16 throughout,
//! static shapes, one 30 s window). The math follows gpu/engine.zig (and
//! npu/net.zig) exactly; the tensors are laid out the same way as the NPU
//! graphs.
//!
//! Encoder ("stt_encoder"):
//!   mel [1][n_mels][3000] fp16   log-mel from the host (frontend.zig)
//!   -> conv1 (k3, pad 1) + GELU -> conv2 (k3, stride 2, pad 1) + GELU
//!   -> [1][1500][state] + positions -> blocks -> ln_post
//!   -> per decoder layer i: ck<i>, cv<i> [1][heads][1500][64] (cross K/V)
//!
//! Decoder step ("stt_decoder"), one token, KV cache passed in/out:
//!   x [1][state]                 token + position embedding (host: one row add)
//!   mask [1][1][1][ctx+1]        0 for cached positions < pos, -30000 otherwise;
//!                                the last entry is this token (always 0)
//!   sk<i>, sv<i> [1][heads][ctx][64]   self-attention cache
//!   ck<i>, cv<i> [1][heads][1500][64]  from the encoder
//!   -> kn<i>, vn<i> [1][state]   this position's K/V (the host writes them
//!                                into the cache), logits [1][n_vocab]
//!
//! The greedy pick (masked log-softmax + arg-max) runs on the host over the
//! fp16 logits, as WhisperKit does: 51864-wide reductions are not what the
//! Neural Engine is for, and it keeps the ban list off the graph.
//! The attention scale 1/8 is folded into the query weights (exact in fp16).
//! A LoRA adapter is merged into the decoder weights when the graph is built
//! (see `mergeAdapter`): a new adapter means a new decoder package to compile.
const std = @import("std");
const Allocator = std.mem.Allocator;
const coreml = @import("coreml");
const mil = coreml.mil;
const model_mod = @import("../model.zig");
const Model = model_mod.Model;
const Linear = model_mod.Linear;
const Norm = model_mod.Norm;
const adapter = @import("../adapter.zig");
const mel = @import("../mel.zig");

pub const head_dim = 64;
pub const mask_off: f32 = -30000.0;
pub const n_frames = mel.n_frames;

pub const Config = struct {
    /// .tanh_approximation is what the GPU computes; .exact (erf) is what
    /// the model was trained with.
    gelu: mil.GeluMode = .tanh_approximation,
    target: mil.Target = .ios17,
};

/// Decoder weights of the adapted layers: base, or with a LoRA folded in.
pub const DecWeights = struct {
    /// [layer][slot] -> f16 [n_out][n_in]
    w: []const [adapter.n_slots][]const f16,
};

/// Linear layer with weights `w` (f16 [out][in]) scaled by `scale`.
fn linear(b: *mil.Builder, x: mil.V, l: Linear, w: []const f16, scale: f32) !mil.V {
    const wt = if (scale == 1) w else blk: {
        const s = try b.arena.alloc(f16, w.len);
        for (s, w) |*o, v| o.* = @floatCast(@as(f32, @floatCast(v)) * scale);
        break :blk s;
    };
    const wv = try b.constF16(&.{ l.n_out, l.n_in }, wt);
    const bias = if (l.b) |bv| blk: {
        const s = try b.arena.alloc(f32, bv.len);
        for (s, bv) |*o, v| o.* = v * scale;
        break :blk try b.constFloat(.fp16, &.{l.n_out}, s);
    } else null;
    return b.linear(x, wv, bias);
}

fn norm(b: *mil.Builder, x: mil.V, n: Norm) !mil.V {
    return b.layerNorm(x, try b.constFloat(.fp16, &.{n.w.len}, n.w), try b.constFloat(.fp16, &.{n.b.len}, n.b), 1e-5);
}

/// [1][n][heads*64] -> [1][heads][n][64]
fn splitHeads(b: *mil.Builder, x: mil.V, heads: usize) !mil.V {
    const n = b.shape(x)[1];
    return b.transpose(try b.reshape(x, &.{ 1, n, heads, head_dim }), &.{ 0, 2, 1, 3 });
}

/// [1][heads][n][64] -> [1][n][heads*64]
fn mergeHeads(b: *mil.Builder, x: mil.V) !mil.V {
    const s = b.shape(x);
    return b.reshape(try b.transpose(x, &.{ 0, 2, 1, 3 }), &.{ 1, s[2], s[1] * head_dim });
}

/// Builds the encoder. Input: mel. Outputs: ck0, cv0, ck1, cv1, ...
pub fn encoder(b: *mil.Builder, m: *const Model, cfg: Config) !void {
    const hp = m.hp;
    const ns = hp.n_audio_state;
    const heads = hp.n_audio_head;
    const A = hp.n_audio_ctx;
    var x = try b.input("mel", .fp16, &.{ 1, hp.n_mels, n_frames });
    // Conv1d weights [out][in][3]: the model's [out][in*3] rows, as stored.
    const w1 = try b.constF16(&.{ ns, hp.n_mels, 3 }, m.conv1.w);
    x = try b.gelu(try b.conv1d(x, w1, try b.constFloat(.fp16, &.{ns}, m.conv1.b.?), 1, .{ 1, 1 }), cfg.gelu);
    const w2 = try b.constF16(&.{ ns, ns, 3 }, m.conv2.w);
    x = try b.gelu(try b.conv1d(x, w2, try b.constFloat(.fp16, &.{ns}, m.conv2.b.?), 2, .{ 1, 1 }), cfg.gelu);
    x = try b.transpose(x, &.{ 0, 2, 1 }); // [1][1500][state]
    x = try b.add(x, try b.constFloat(.fp16, &.{ 1, A, ns }, m.enc_pos));

    const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
    for (m.enc_layers) |l| {
        var h = try norm(b, x, l.attn_ln);
        const q = try splitHeads(b, try linear(b, h, l.attn.q, l.attn.q.w, scale), heads);
        const k = try splitHeads(b, try linear(b, h, l.attn.k, l.attn.k.w, 1), heads);
        const v = try splitHeads(b, try linear(b, h, l.attn.v, l.attn.v.w, 1), heads);
        const p = try b.softmax(try b.matmul(q, k, false, true), -1); // [1][H][A][A]
        const o = try mergeHeads(b, try b.matmul(p, v, false, false));
        x = try b.add(x, try linear(b, o, l.attn.o, l.attn.o.w, 1));
        h = try norm(b, x, l.mlp_ln);
        const f = try b.gelu(try linear(b, h, l.fc1, l.fc1.w, 1), cfg.gelu);
        x = try b.add(x, try linear(b, f, l.fc2, l.fc2.w, 1));
    }
    x = try norm(b, x, m.enc_ln_post);

    const th = hp.n_text_head;
    var name_buf: [16]u8 = undefined;
    for (m.dec_layers, 0..) |l, i| {
        try b.output(try std.fmt.bufPrint(&name_buf, "ck{d}", .{i}), try splitHeads(b, try linear(b, x, l.cross.k, l.cross.k.w, 1), th));
        try b.output(try std.fmt.bufPrint(&name_buf, "cv{d}", .{i}), try splitHeads(b, try linear(b, x, l.cross.v, l.cross.v.w, 1), th));
    }
}

/// Builds the decoder step. Inputs: x, mask, then per layer sk, sv, ck, cv.
/// Outputs: per layer kn, vn, then logits.
pub fn decoder(b: *mil.Builder, m: *const Model, cfg: Config, dw: DecWeights) !void {
    const hp = m.hp;
    const A = hp.n_audio_ctx;
    const ts = hp.n_text_state;
    const heads = hp.n_text_head;
    const ctx = hp.n_text_ctx;
    var x = try b.input("x", .fp16, &.{ 1, ts });
    const mask = try b.input("mask", .fp16, &.{ 1, 1, 1, ctx + 1 });
    const L = hp.n_text_layer;
    var name_buf: [16]u8 = undefined;
    const sk = try b.arena.alloc(mil.V, L);
    const sv = try b.arena.alloc(mil.V, L);
    const ck = try b.arena.alloc(mil.V, L);
    const cv = try b.arena.alloc(mil.V, L);
    for (0..L) |i| {
        sk[i] = try b.input(try std.fmt.bufPrint(&name_buf, "sk{d}", .{i}), .fp16, &.{ 1, heads, ctx, head_dim });
        sv[i] = try b.input(try std.fmt.bufPrint(&name_buf, "sv{d}", .{i}), .fp16, &.{ 1, heads, ctx, head_dim });
        ck[i] = try b.input(try std.fmt.bufPrint(&name_buf, "ck{d}", .{i}), .fp16, &.{ 1, heads, A, head_dim });
        cv[i] = try b.input(try std.fmt.bufPrint(&name_buf, "cv{d}", .{i}), .fp16, &.{ 1, heads, A, head_dim });
    }

    const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
    const one: []const usize = &.{ 1, heads, 1, head_dim };
    for (m.dec_layers, 0..) |l, i| {
        const w = dw.w[i];
        // self-attention over the cache plus this token
        var h = try norm(b, x, l.attn_ln);
        const q = try linear(b, h, l.attn.q, w[@intFromEnum(adapter.Slot.q)], scale);
        const k = try linear(b, h, l.attn.k, w[@intFromEnum(adapter.Slot.k)], 1);
        const v = try linear(b, h, l.attn.v, w[@intFromEnum(adapter.Slot.v)], 1);
        try b.output(try std.fmt.bufPrint(&name_buf, "kn{d}", .{i}), k);
        try b.output(try std.fmt.bufPrint(&name_buf, "vn{d}", .{i}), v);
        const kk = try b.concat(&.{ sk[i], try b.reshape(k, one) }, 2); // [1][H][ctx+1][64]
        const vv = try b.concat(&.{ sv[i], try b.reshape(v, one) }, 2);
        const s = try b.add(try b.matmul(try b.reshape(q, one), kk, false, true), mask); // [1][H][1][ctx+1]
        var o = try b.matmul(try b.softmax(s, -1), vv, false, false); // [1][H][1][64]
        x = try b.add(x, try linear(b, try b.reshape(o, &.{ 1, ts }), l.attn.o, w[@intFromEnum(adapter.Slot.o)], 1));

        // cross-attention over the audio
        h = try norm(b, x, l.cross_ln);
        const cq = try linear(b, h, l.cross.q, w[@intFromEnum(adapter.Slot.cross_q)], scale);
        const cs = try b.matmul(try b.reshape(cq, one), ck[i], false, true); // [1][H][1][A]
        o = try b.matmul(try b.softmax(cs, -1), cv[i], false, false);
        x = try b.add(x, try linear(b, try b.reshape(o, &.{ 1, ts }), l.cross.o, w[@intFromEnum(adapter.Slot.cross_o)], 1));

        h = try norm(b, x, l.mlp_ln);
        const f = try b.gelu(try linear(b, h, l.fc1, w[@intFromEnum(adapter.Slot.fc1)], 1), cfg.gelu);
        x = try b.add(x, try linear(b, f, l.fc2, w[@intFromEnum(adapter.Slot.fc2)], 1));
    }
    x = try norm(b, x, m.dec_ln);
    // The token table is also the output projection.
    const emb = try b.constF16(&.{ hp.n_vocab, ts }, m.tok_emb);
    try b.output("logits", try b.linear(x, emb, null));
}

/// Base decoder weights (no adapter): slices of the model.
pub fn baseWeights(gpa: Allocator, m: *const Model) ![][adapter.n_slots][]const f16 {
    const L = m.hp.n_text_layer;
    const slots = try gpa.alloc([adapter.n_slots][]const f16, L);
    for (0..L) |l| for (0..adapter.n_slots) |si| {
        slots[l][si] = adapter.slotLinear(m, l, @enumFromInt(si)).w;
    };
    return slots;
}

/// The adapted decoder weights W + scale·B·A (f16), same arithmetic as
/// kernels.lora_merge and the NPU engine. Returns the slot table and the
/// buffer that owns the merged weights (free both).
pub fn mergeAdapter(gpa: Allocator, m: *const Model, a: *const adapter.Adapter) !struct { slots: [][adapter.n_slots][]const f16, buf: []f16 } {
    const L = m.hp.n_text_layer;
    var total: usize = 0;
    for (0..L) |l| for (0..adapter.n_slots) |si| {
        total += adapter.slotLinear(m, l, @enumFromInt(si)).w.len;
    };
    const buf = try gpa.alloc(f16, total);
    errdefer gpa.free(buf);
    const slots = try gpa.alloc([adapter.n_slots][]const f16, L);
    var off: usize = 0;
    const scale = a.scale();
    for (0..L) |l| for (0..adapter.n_slots) |si| {
        const s: adapter.Slot = @enumFromInt(si);
        const lin = adapter.slotLinear(m, l, s);
        const lo = a.get(l, s);
        const w = buf[off..][0..lin.w.len];
        off += lin.w.len;
        for (0..lin.n_out) |o| {
            const brow = lo.b[o * a.rank ..][0..a.rank];
            for (0..lin.n_in) |i| {
                var d: f32 = 0;
                for (brow, 0..) |bv, r| d += bv * lo.a[r * lin.n_in + i];
                const idx = o * lin.n_in + i;
                w[idx] = @floatCast(@as(f32, @floatCast(lin.w[idx])) + scale * d);
            }
        }
        slots[l][si] = w;
    };
    return .{ .slots = slots, .buf = buf };
}
