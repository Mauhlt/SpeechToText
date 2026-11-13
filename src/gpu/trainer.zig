//! Personal fine-tuning on the GPU: LoRA adapters on every decoder linear
//! layer, trained with a hand-written backward pass and Muon (or Adam) on
//! the user's own (audio, corrected text) samples. The base weights and the
//! encoder stay frozen; only the small low-rank matrices learn.
//!
//! Everything runs on the device: the encoder pass per sample, teacher-forced
//! decoder forward with saved activations, backward, gradient clipping and the
//! optimizer. The host only uploads each sample's audio and tokens and reads
//! back the loss once per epoch and the adapter at the end.
//!
//! Uses its own `Engine` (a second copy of the weights in VRAM, without any
//! adapter merged in), so it can run on a background thread while another
//! engine keeps transcribing. Both share the GPU, so each runs slower.
const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("cuda.zig");
const eng = @import("engine.zig");
const Engine = eng.Engine;
const DevPtr = eng.DevPtr;
const DLinear = eng.DLinear;
const Operand = eng.Operand;
const flag_bias = eng.flag_bias;
const flag_accum = eng.flag_accum;
const head_dim = eng.head_dim;
const Model = @import("../model.zig").Model;
const Tokenizer = @import("../bpe.zig").Tokenizer;
const adapter = @import("../adapter.zig");
const Adapter = adapter.Adapter;
const Slot = adapter.Slot;
const n_slots = adapter.n_slots;

pub const Sample = struct { audio: []const f32, text: []const u8 };

pub const Optimizer = enum { muon, adam };

pub const Options = struct {
    epochs: usize = 3,
    /// null = the optimizer's default (tuned on LoRA: muon 4e-3, adam 1e-3).
    lr: ?f32 = null,
    optimizer: Optimizer = .muon,
    batch: usize = 4,
    clip_norm: f32 = 1.0,
    seed: u64 = 1,
};

/// Reported after every optimizer step (and with the loss after each epoch).
pub const Step = struct {
    epoch: usize,
    epochs: usize,
    /// Samples processed so far, out of `total` (= samples × epochs).
    done: usize,
    total: usize,
    /// Mean loss of the last finished epoch (NaN before the first).
    loss: f32,
    /// True for the report sent when an epoch finishes (`loss` is that epoch's).
    epoch_done: bool = false,
};

pub const Progress = struct {
    ctx: *anyopaque,
    onStep: *const fn (ctx: *anyopaque, s: Step) void,
};

/// Fine-tunes `ad` on `samples` on the GPU. Returns the final epoch's mean
/// loss. If `cancel` becomes true, stops between samples with error.Canceled
/// (`ad` is then left unchanged).
pub fn train(
    gpa: Allocator,
    model: *const Model,
    ad: *Adapter,
    samples: []const Sample,
    opts: Options,
    progress: ?Progress,
    cancel: ?*const std.atomic.Value(bool),
) !f32 {
    if (samples.len == 0) return error.NoSamples;
    var t = try Trainer.init(gpa, model, ad, samples, opts.optimizer);
    defer t.deinit();

    const order = try gpa.alloc(usize, samples.len);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = i;
    var rng: std.Random.DefaultPrng = .init(opts.seed);
    const lr: f32 = opts.lr orelse switch (opts.optimizer) {
        .muon => @as(f32, 4e-3),
        .adam => @as(f32, 1e-3),
    };

    var last_loss = std.math.nan(f32);
    const total = samples.len * opts.epochs;
    var done: usize = 0;
    for (0..opts.epochs) |epoch| {
        rng.random().shuffle(usize, order);
        try t.e.gpu.memset32(t.loss, 0, 1);
        var in_batch: usize = 0;
        for (order, 0..) |idx, k| {
            if (cancel) |c| if (c.load(.acquire)) return error.Canceled;
            try t.sample(samples[idx].audio, idx);
            in_batch += 1;
            done += 1;
            if (in_batch == opts.batch or k == order.len - 1) {
                try t.optimize(in_batch, opts, lr);
                in_batch = 0;
                if (progress) |p| p.onStep(p.ctx, .{ .epoch = epoch + 1, .epochs = opts.epochs, .done = done, .total = total, .loss = last_loss });
            }
        }
        last_loss = try t.readLoss() / @as(f32, @floatFromInt(samples.len));
        if (progress) |p| p.onStep(p.ctx, .{ .epoch = epoch + 1, .epochs = opts.epochs, .done = done, .total = total, .loss = last_loss, .epoch_done = true });
    }
    try t.download(ad);
    return last_loss;
}

/// Mean per-token cross-entropy of `s.text` given `s.audio` under the base
/// model + `ad` (no gradient). Lower is better.
pub fn evalLoss(gpa: Allocator, model: *const Model, ad: *Adapter, s: Sample) !f32 {
    var t = try Trainer.init(gpa, model, ad, &.{s}, .adam);
    defer t.deinit();
    try t.e.gpu.memset32(t.loss, 0, 1);
    try t.e.encode(s.audio);
    try t.forwardBackward(0, false);
    return t.readLoss();
}

/// Loss and full adapter gradient for one sample (for tests and gradient
/// checks): writes d(loss)/d(ad.buf) into `grad` (same layout as `ad.buf`).
pub fn lossAndGrad(gpa: Allocator, model: *const Model, ad: *Adapter, s: Sample, grad: []f32) !f32 {
    std.debug.assert(grad.len == ad.buf.len);
    var t = try Trainer.init(gpa, model, ad, &.{s}, .adam);
    defer t.deinit();
    try t.e.gpu.memset32(t.loss, 0, 1);
    try t.e.encode(s.audio);
    try t.forwardBackward(0, true);
    const g = &t.e.gpu;
    const stage = try g.allocPinned(std.mem.alignForward(usize, grad.len * 4, 4096));
    defer g.freePinned(stage);
    try g.download(stage[0 .. grad.len * 4], t.grads);
    try g.sync();
    @memcpy(std.mem.sliceAsBytes(grad), stage[0 .. grad.len * 4]);
    return t.readLoss();
}

/// Per-layer saved activations (device pointers, f32).
const Acts = struct {
    h1: DevPtr,
    q: DevPtr,
    k: DevPtr,
    v: DevPtr,
    p: DevPtr, // [H][L][L]
    a: DevPtr,
    x1: DevPtr,
    h2: DevPtr,
    qc: DevPtr,
    pc: DevPtr, // [H][L][A]
    ac: DevPtr,
    x2: DevPtr,
    h3: DevPtr,
    u: DevPtr, // pre-GELU [L][4s]
    g: DevPtr, // post-GELU [L][4s]
    mean: [3]DevPtr,
    rstd: [3]DevPtr,
    z: [n_slots]DevPtr, // LoRA intermediates x·Aᵀ, [L][rank]
};

const Seq = struct { tokens: DevPtr, targets: DevPtr, len: u32, first: u32, n_targets: u32 };

const Trainer = struct {
    gpa: Allocator,
    e: *Engine,
    model: *const Model,
    ad: *const Adapter,
    rank: u32,
    n_params: usize,
    // Device state.
    mem: DevPtr,
    params: DevPtr,
    grads: DevPtr,
    opt1: DevPtr, // muon momentum / adam m
    opt2: DevPtr, // muon scratch / adam v
    desc: DevPtr, // muon matrix table
    n_mats: u32,
    loss: DevPtr,
    sq: DevPtr,
    seqs: []Seq,
    l_max: u32,
    xs: []DevPtr, // layer inputs [nl + 1][L][s]
    acts: []Acts,
    hf: DevPtr,
    lnf_mean: DevPtr,
    lnf_rstd: DevPtr,
    logits: DevPtr,
    tmp: DevPtr, // [L][4s]
    dx: DevPtr,
    dhf: DevPtr,
    dh: DevPtr, // [L][4s]
    dq: DevPtr,
    dk: DevPtr,
    dv: DevPtr,
    da: DevPtr,
    dh_in: DevPtr,
    dyb: DevPtr, // [L][rank]
    dp: DevPtr, // [H][L][max(L, A)]
    adam_t: u32 = 0,

    fn init(gpa: Allocator, model: *const Model, ad: *const Adapter, samples: []const Sample, optimizer: Optimizer) !Trainer {
        if (ad.rank > Adapter.max_rank) return error.AdapterRankTooLarge;
        const hp = model.hp;
        const e = try Engine.init(gpa, model); // base weights, no adapter merged
        errdefer e.deinit(gpa);
        const g = &e.gpu;

        // Tokenize every sample: input = prefix + text, targets = text + eot.
        var tok = try Tokenizer.init(gpa, model);
        defer tok.deinit(gpa);
        const tk = model.tokens;
        var prefix_buf: [3]u32 = undefined;
        var prefix: std.ArrayList(u32) = .initBuffer(&prefix_buf);
        prefix.appendAssumeCapacity(tk.sot);
        if (tk.multilingual) {
            prefix.appendAssumeCapacity(tk.lang_en);
            prefix.appendAssumeCapacity(tk.transcribe);
        }
        prefix.appendAssumeCapacity(tk.not);
        const pl = prefix.items.len;
        const max_text = hp.n_text_ctx - pl - 1;

        const seqs = try gpa.alloc(Seq, samples.len);
        errdefer gpa.free(seqs);
        var host: std.ArrayList(u32) = .empty; // all token + target arrays, back to back
        defer host.deinit(gpa);
        var l_max: u32 = 1;
        const offsets = try gpa.alloc([2]usize, samples.len);
        defer gpa.free(offsets);
        for (samples, 0..) |s, i| {
            const spaced = try std.mem.concat(gpa, u8, &.{ " ", std.mem.trim(u8, s.text, " \t\r\n") });
            defer gpa.free(spaced);
            const toks = try tok.encode(gpa, spaced);
            defer gpa.free(toks);
            const text = toks[0..@min(toks.len, max_text)];
            const L = pl + text.len;
            const first = pl - 1; // first position with a target
            // If the text was cut to fit the context, the last position's true
            // next token is unknown, so it gets no target (predicting
            // end-of-text there would teach the model to stop early).
            const truncated = toks.len > max_text;
            const n_t = L - first - @intFromBool(truncated);
            offsets[i][0] = host.items.len;
            try host.appendSlice(gpa, prefix.items);
            try host.appendSlice(gpa, text);
            offsets[i][1] = host.items.len;
            for (0..n_t) |t| try host.append(gpa, if (first + t + 1 < L) (if (first + t + 1 < pl) prefix.items[first + t + 1] else text[first + t + 1 - pl]) else tk.eot);
            seqs[i] = .{ .tokens = 0, .targets = 0, .len = @intCast(L), .first = @intCast(first), .n_targets = @intCast(n_t) };
            l_max = @max(l_max, @as(u32, @intCast(L)));
        }

        // Device memory: one allocation, carved up below.
        const s = hp.n_text_state;
        const H = hp.n_text_head;
        const A = hp.n_audio_ctx;
        const nl = hp.n_text_layer;
        const L: usize = l_max;
        const r = ad.rank;
        const np = ad.buf.len;
        const n_mats: u32 = @intCast(2 * ad.loras.len);
        var plan: Carver = .{};
        var t: Trainer = undefined;
        t.params = plan.take(np);
        t.grads = plan.take(np);
        t.opt1 = plan.take(np);
        t.opt2 = plan.take(np);
        t.desc = plan.take(3 * n_mats);
        t.loss = plan.take(1);
        t.sq = plan.take(1);
        const tok_mem = plan.take(host.items.len);
        const xs_base = plan.take((nl + 1) * L * s);
        var probe: Carver = .{};
        _ = carveLayer(&probe, 0, L, s, H, A, r);
        const layer_bytes = probe.off;
        const acts_base = plan.takeBytes(nl * layer_bytes);
        t.hf = plan.take(L * s);
        t.lnf_mean = plan.take(L);
        t.lnf_rstd = plan.take(L);
        t.logits = plan.take(L * hp.n_vocab);
        t.tmp = plan.take(L * 4 * s);
        t.dx = plan.take(L * s);
        t.dhf = plan.take(L * s);
        t.dh = plan.take(L * 4 * s);
        t.dq = plan.take(L * s);
        t.dk = plan.take(L * s);
        t.dv = plan.take(L * s);
        t.da = plan.take(L * s);
        t.dh_in = plan.take(L * s);
        t.dyb = plan.take(L * r);
        t.dp = plan.take(H * L * @max(L, A));

        const mem = g.alloc(plan.off) catch |err| {
            std.log.err("training needs {d} MB of GPU memory for samples up to {d} tokens", .{ plan.off >> 20, L });
            return err;
        };
        errdefer g.free(mem);
        // Rebase every pointer onto the allocation.
        inline for (.{ "params", "grads", "opt1", "opt2", "desc", "loss", "sq", "hf", "lnf_mean", "lnf_rstd", "logits", "tmp", "dx", "dhf", "dh", "dq", "dk", "dv", "da", "dh_in", "dyb", "dp" }) |f| @field(t, f) += mem;

        const xs = try gpa.alloc(DevPtr, nl + 1);
        errdefer gpa.free(xs);
        for (xs, 0..) |*x, i| x.* = mem + xs_base + i * L * s * 4;
        const acts = try gpa.alloc(Acts, nl);
        errdefer gpa.free(acts);
        for (acts, 0..) |*act, li| {
            var c: Carver = .{ .off = acts_base + li * layer_bytes };
            act.* = carveLayer(&c, mem, L, s, H, A, r);
        }

        // Upload parameters, tokens and the Muon matrix table; zero the rest.
        const desc = try gpa.alloc(u32, 3 * n_mats);
        defer gpa.free(desc);
        var mi: usize = 0;
        for (0..nl) |li| for (0..n_slots) |si| {
            const sl: Slot = @enumFromInt(si);
            const lo = ad.get(li, sl);
            const off = ad.offsetOf(li, sl);
            desc[mi * 3 ..][0..3].* = .{ @intCast(off), @intCast(r), @intCast(lo.n_in) };
            desc[mi * 3 + 3 ..][0..3].* = .{ @intCast(off + lo.a.len), @intCast(lo.n_out), @intCast(r) };
            mi += 2;
        };
        {
            var up: eng.Uploader = .{ .gpu = g };
            defer up.end();
            try up.begin();
            try up.putAt(t.params, f32, ad.buf);
            try up.putAt(t.desc, u32, desc);
            try up.putAt(mem + tok_mem, u32, host.items);
            try g.memset32(t.grads, 0, np);
            try g.memset32(t.opt1, 0, np);
            try g.memset32(t.opt2, 0, np);
            try g.sync();
        }
        for (seqs, offsets) |*sq, o| {
            sq.tokens = mem + tok_mem + o[0] * 4;
            sq.targets = mem + tok_mem + o[1] * 4;
        }
        _ = optimizer;

        t.gpa = gpa;
        t.e = e;
        t.model = model;
        t.ad = ad;
        t.rank = @intCast(r);
        t.n_params = np;
        t.mem = mem;
        t.n_mats = n_mats;
        t.seqs = seqs;
        t.l_max = l_max;
        t.xs = xs;
        t.acts = acts;
        t.adam_t = 0;
        return t;
    }

    fn deinit(t: *Trainer) void {
        t.e.gpu.makeCurrent();
        t.e.gpu.sync() catch {};
        t.e.gpu.free(t.mem);
        t.gpa.free(t.seqs);
        t.gpa.free(t.xs);
        t.gpa.free(t.acts);
        t.e.deinit(t.gpa);
    }

    fn readLoss(t: *Trainer) !f32 {
        var v: [1]f32 = undefined;
        const pinned = t.e.pinned[t.e.pinned.len - 16 ..][0..4];
        try t.e.gpu.download(pinned, t.loss);
        try t.e.gpu.sync();
        @memcpy(std.mem.asBytes(&v), pinned);
        return v[0];
    }

    fn download(t: *Trainer, ad: *Adapter) !void {
        const g = &t.e.gpu;
        const stage = try g.allocPinned(std.mem.alignForward(usize, t.n_params * 4, 4096));
        defer g.freePinned(stage);
        try g.download(stage[0 .. t.n_params * 4], t.params);
        try g.sync();
        @memcpy(std.mem.sliceAsBytes(ad.buf), stage[0 .. t.n_params * 4]);
    }

    fn sample(t: *Trainer, audio: []const f32, idx: usize) !void {
        try t.e.encode(audio);
        try t.forwardBackward(idx, true);
    }

    // -------------------------------------------------------- building blocks

    fn mm(t: *Trainer, comptime w16: bool, out: Operand, a: Operand, w: Operand, bias: DevPtr, m: u32, n: u32, k: u32, flags: u32, alpha: f32, batch: u32) !void {
        try t.e.mm(w16, out, a, w, bias, m, n, k, flags, alpha, batch);
    }

    fn loraA(t: *const Trainer, base: DevPtr, li: usize, s: Slot) DevPtr {
        return base + t.ad.offsetOf(li, s) * 4;
    }
    fn loraB(t: *const Trainer, base: DevPtr, li: usize, s: Slot) DevPtr {
        return base + (t.ad.offsetOf(li, s) + t.ad.get(li, s).a.len) * 4;
    }

    /// out (=|+=) x·Wᵀ + b + scale·(x·Aᵀ)·Bᵀ, saving z = x·Aᵀ.
    fn linFwd(t: *Trainer, out: DevPtr, x: DevPtr, L: u32, li: usize, s: Slot, accum: bool) !void {
        const d = t.e.slot(li, s);
        const r = t.rank;
        const z = t.acts[li].z[@intFromEnum(s)];
        const fl: u32 = (if (accum) flag_accum else 0) | (if (d.b != 0) flag_bias else 0);
        try t.mm(true, .{ .ptr = out, .i = d.n_out, .p = 1 }, .{ .ptr = x, .i = d.n_in, .p = 1 }, .{ .ptr = d.w, .i = d.n_in, .p = 1 }, d.b, L, d.n_out, d.n_in, fl, 1, 1);
        try t.mm(false, .{ .ptr = z, .i = r, .p = 1 }, .{ .ptr = x, .i = d.n_in, .p = 1 }, .{ .ptr = t.loraA(t.params, li, s), .i = d.n_in, .p = 1 }, 0, L, r, d.n_in, 0, 1, 1);
        try t.mm(false, .{ .ptr = out, .i = d.n_out, .p = 1 }, .{ .ptr = z, .i = r, .p = 1 }, .{ .ptr = t.loraB(t.params, li, s), .i = r, .p = 1 }, 0, L, d.n_out, r, flag_accum, t.ad.scale(), 1);
    }

    /// Given dy, writes (or adds to) dx and accumulates the LoRA gradients.
    fn linBack(t: *Trainer, dx: DevPtr, dy: DevPtr, x: DevPtr, L: u32, li: usize, s: Slot, accum: bool) !void {
        const d = t.e.slot(li, s);
        const r = t.rank;
        const sc = t.ad.scale();
        const z = t.acts[li].z[@intFromEnum(s)];
        const A = t.loraA(t.params, li, s);
        const B = t.loraB(t.params, li, s);
        // dx = dy·W + scale·(dy·B)·A
        try t.mm(true, .{ .ptr = dx, .i = d.n_in, .p = 1 }, .{ .ptr = dy, .i = d.n_out, .p = 1 }, .{ .ptr = d.w, .i = 1, .p = d.n_in }, 0, L, d.n_in, d.n_out, if (accum) flag_accum else 0, 1, 1);
        try t.mm(false, .{ .ptr = t.dyb, .i = r, .p = 1 }, .{ .ptr = dy, .i = d.n_out, .p = 1 }, .{ .ptr = B, .i = 1, .p = r }, 0, L, r, d.n_out, 0, 1, 1);
        try t.mm(false, .{ .ptr = dx, .i = d.n_in, .p = 1 }, .{ .ptr = t.dyb, .i = r, .p = 1 }, .{ .ptr = A, .i = 1, .p = d.n_in }, 0, L, d.n_in, r, flag_accum, sc, 1);
        // dB += scale·dyᵀ·z ; dA += scale·(dy·B)ᵀ·x
        try t.mm(false, .{ .ptr = t.loraB(t.grads, li, s), .i = r, .p = 1 }, .{ .ptr = dy, .i = 1, .p = d.n_out }, .{ .ptr = z, .i = 1, .p = r }, 0, d.n_out, r, L, flag_accum, sc, 1);
        try t.mm(false, .{ .ptr = t.loraA(t.grads, li, s), .i = d.n_in, .p = 1 }, .{ .ptr = t.dyb, .i = 1, .p = r }, .{ .ptr = x, .i = 1, .p = d.n_in }, 0, r, d.n_in, L, flag_accum, sc, 1);
    }

    fn lnFwd(t: *Trainer, out: DevPtr, x: DevPtr, L: u32, nm: eng.DNorm, mean: DevPtr, rstd: DevPtr) !void {
        const s: u32 = @intCast(t.model.hp.n_text_state);
        try t.e.gpu.launch(t.e.k.layernorm_fwd, .{ L, 1, 1 }, 256, .{ out, x, nm.w, nm.b, s, mean, rstd });
    }
    fn lnBack(t: *Trainer, dx: DevPtr, dy: DevPtr, x: DevPtr, L: u32, nm: eng.DNorm, mean: DevPtr, rstd: DevPtr) !void {
        const s: u32 = @intCast(t.model.hp.n_text_state);
        try t.e.gpu.launch(t.e.k.layernorm_back, .{ L, 1, 1 }, 256, .{ dx, dy, x, nm.w, mean, rstd, s });
    }

    /// Per-head views: q/k/v/out are [rows][s] with head h at columns h*64..;
    /// p is [H][lq][lkv].
    fn qkv(ptr: DevPtr, s: u32) Operand {
        return .{ .ptr = ptr, .i = s, .p = 1, .b = head_dim };
    }

    fn attnFwd(t: *Trainer, out: DevPtr, q: DevPtr, k: DevPtr, v: DevPtr, p: DevPtr, lq: u32, lkv: u32, causal: bool) !void {
        const hp = t.model.hp;
        const s: u32 = @intCast(hp.n_text_state);
        const H: u32 = @intCast(hp.n_text_head);
        const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
        const pv: Operand = .{ .ptr = p, .i = lkv, .p = 1, .b = lq * lkv };
        // P = softmax(scale · Q·Kᵀ)
        try t.mm(false, pv, qkv(q, s), .{ .ptr = k, .i = s, .p = 1, .b = head_dim }, 0, lq, lkv, head_dim, 0, scale, H);
        try t.e.gpu.launch(t.e.k.softmax_rows, .{ H * lq, 1, 1 }, 256, .{ p, lkv, lq, @as(u32, @intFromBool(causal)) });
        // out = P·V
        try t.mm(false, qkv(out, s), pv, .{ .ptr = v, .i = 1, .p = s, .b = head_dim }, 0, lq, head_dim, lkv, 0, 1, H);
    }

    /// Writes dq (and dk, dv for self-attention) from da.
    fn attnBack(t: *Trainer, dq: DevPtr, dk: ?DevPtr, dv: ?DevPtr, da: DevPtr, q: DevPtr, k: DevPtr, v: DevPtr, p: DevPtr, lq: u32, lkv: u32) !void {
        const hp = t.model.hp;
        const s: u32 = @intCast(hp.n_text_state);
        const H: u32 = @intCast(hp.n_text_head);
        const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
        const ds: Operand = .{ .ptr = t.dp, .i = lkv, .p = 1, .b = lq * lkv };
        // dP = dA·Vᵀ, then dS = softmax'(dP)·scale
        try t.mm(false, ds, qkv(da, s), .{ .ptr = v, .i = s, .p = 1, .b = head_dim }, 0, lq, lkv, head_dim, 0, 1, H);
        try t.e.gpu.launch(t.e.k.softmax_back, .{ H * lq, 1, 1 }, 256, .{ t.dp, p, lkv, scale });
        // dQ = dS·K
        try t.mm(false, qkv(dq, s), ds, .{ .ptr = k, .i = 1, .p = s, .b = head_dim }, 0, lq, head_dim, lkv, 0, 1, H);
        // dK = dSᵀ·Q ; dV = Pᵀ·dA
        const dsT: Operand = .{ .ptr = t.dp, .i = 1, .p = lkv, .b = lq * lkv };
        const pT: Operand = .{ .ptr = p, .i = 1, .p = lkv, .b = lq * lkv };
        if (dk) |x| try t.mm(false, qkv(x, s), dsT, .{ .ptr = q, .i = 1, .p = s, .b = head_dim }, 0, lkv, head_dim, lq, 0, 1, H);
        if (dv) |x| try t.mm(false, qkv(x, s), pT, .{ .ptr = da, .i = 1, .p = s, .b = head_dim }, 0, lkv, head_dim, lq, 0, 1, H);
    }

    // -------------------------------------------------------- forward / backward

    /// Teacher-forced forward pass (+ backward if `backward`). Adds the
    /// sample's mean token loss to `loss` and its gradients to `grads`.
    fn forwardBackward(t: *Trainer, idx: usize, backward: bool) !void {
        const e = t.e;
        const g = &e.gpu;
        const hp = t.model.hp;
        const sq = t.seqs[idx];
        const L = sq.len;
        const s: u32 = @intCast(hp.n_text_state);
        const A: u32 = @intCast(hp.n_audio_ctx);
        const V: u32 = @intCast(hp.n_vocab);
        const n = L * s * 4; // bytes of one [L][s] activation

        try g.launch(e.k.embed_seq, .{ (s + 255) / 256, L, 1 }, 256, .{ t.xs[0], e.tok_emb, e.dec_pos, sq.tokens, s });
        for (e.dec, 0..) |l, li| {
            const act = t.acts[li];
            const x0 = t.xs[li];
            try t.lnFwd(act.h1, x0, L, l.attn_ln, act.mean[0], act.rstd[0]);
            try t.linFwd(act.q, act.h1, L, li, .q, false);
            try t.linFwd(act.k, act.h1, L, li, .k, false);
            try t.linFwd(act.v, act.h1, L, li, .v, false);
            try t.attnFwd(act.a, act.q, act.k, act.v, act.p, L, L, true);
            try g.copy(act.x1, x0, n);
            try t.linFwd(act.x1, act.a, L, li, .o, true);

            const kv = e.crossKV(li);
            try t.lnFwd(act.h2, act.x1, L, l.cross_ln, act.mean[1], act.rstd[1]);
            try t.linFwd(act.qc, act.h2, L, li, .cross_q, false);
            try t.attnFwd(act.ac, act.qc, kv.k, kv.v, act.pc, L, A, false);
            try g.copy(act.x2, act.x1, n);
            try t.linFwd(act.x2, act.ac, L, li, .cross_o, true);

            try t.lnFwd(act.h3, act.x2, L, l.mlp_ln, act.mean[2], act.rstd[2]);
            try t.linFwd(act.u, act.h3, L, li, .fc1, false);
            try g.launch(e.k.gelu_fwd, .{ (L * 4 * s + 255) / 256, 1, 1 }, 256, .{ act.g, act.u, L * 4 * s });
            try g.copy(t.xs[li + 1], act.x2, n);
            try t.linFwd(t.xs[li + 1], act.g, L, li, .fc2, true);
        }
        const xf = t.xs[e.dec.len];
        try t.lnFwd(t.hf, xf, L, e.dec_ln, t.lnf_mean, t.lnf_rstd);

        // Logits only where there is a target; xent turns them into gradients.
        const nt = sq.n_targets;
        try t.mm(true, .{ .ptr = t.logits, .i = V, .p = 1 }, .{ .ptr = t.hf + sq.first * s * 4, .i = s, .p = 1 }, .{ .ptr = e.tok_emb, .i = s, .p = 1 }, 0, nt, V, s, 0, 1, 1);
        try g.launch(e.k.xent, .{ nt, 1, 1 }, 1024, .{ t.logits, sq.targets, V, t.loss, 1.0 / @as(f32, @floatFromInt(nt)) });
        if (!backward) return;

        // ---- backward ----
        try g.memset32(t.dhf, 0, L * s);
        try g.memset32(t.dx, 0, L * s);
        try t.mm(true, .{ .ptr = t.dhf + sq.first * s * 4, .i = s, .p = 1 }, .{ .ptr = t.logits, .i = V, .p = 1 }, .{ .ptr = e.tok_emb, .i = 1, .p = s }, 0, nt, s, V, 0, 1, 1);
        try t.lnBack(t.dx, t.dhf, xf, L, e.dec_ln, t.lnf_mean, t.lnf_rstd);

        var li = e.dec.len;
        while (li > 0) {
            li -= 1;
            const l = e.dec[li];
            const act = t.acts[li];
            const kv = e.crossKV(li);

            // MLP: x3 = x2 + fc2(gelu(fc1(ln3(x2))))
            try t.linBack(t.dh, t.dx, act.g, L, li, .fc2, false);
            try g.launch(e.k.gelu_back, .{ (L * 4 * s + 255) / 256, 1, 1 }, 256, .{ t.dh, act.u, L * 4 * s });
            try t.linBack(t.dh_in, t.dh, act.h3, L, li, .fc1, false);
            try t.lnBack(t.dx, t.dh_in, act.x2, L, l.mlp_ln, act.mean[2], act.rstd[2]);

            // Cross-attention: x2 = x1 + o(attn(q(ln2(x1)), K, V))
            try t.linBack(t.da, t.dx, act.ac, L, li, .cross_o, false);
            try t.attnBack(t.dq, null, null, t.da, act.qc, kv.k, kv.v, act.pc, L, A);
            try t.linBack(t.dh_in, t.dq, act.h2, L, li, .cross_q, false);
            try t.lnBack(t.dx, t.dh_in, act.x1, L, l.cross_ln, act.mean[1], act.rstd[1]);

            // Self-attention: x1 = x0 + o(attn(q, k, v of ln1(x0)))
            try t.linBack(t.da, t.dx, act.a, L, li, .o, false);
            try t.attnBack(t.dq, t.dk, t.dv, t.da, act.q, act.k, act.v, act.p, L, L);
            try t.linBack(t.dh_in, t.dq, act.h1, L, li, .q, false);
            try t.linBack(t.dh_in, t.dk, act.h1, L, li, .k, true);
            try t.linBack(t.dh_in, t.dv, act.h1, L, li, .v, true);
            try t.lnBack(t.dx, t.dh_in, t.xs[li], L, l.attn_ln, act.mean[0], act.rstd[0]);
        }
    }

    /// Averages the accumulated gradients over the batch, clips them, takes
    /// an optimizer step and zeroes the gradients.
    fn optimize(t: *Trainer, batch: usize, opts: Options, lr: f32) !void {
        const g = &t.e.gpu;
        const k = t.e.k;
        const n: u32 = @intCast(t.n_params);
        const blocks = (n + 255) / 256;
        try g.memset32(t.sq, 0, 1);
        try g.launch(k.sumsq, .{ @min(blocks, 1024), 1, 1 }, 256, .{ t.grads, n, t.sq });
        try g.launch(k.clip_scale, .{ blocks, 1, 1 }, 256, .{ t.grads, n, t.sq, 1.0 / @as(f32, @floatFromInt(batch)), opts.clip_norm });
        switch (opts.optimizer) {
            .muon => try g.launch(k.muon, .{ t.n_mats, 1, 1 }, 256, .{ t.params, t.grads, t.opt1, t.opt2, t.desc, @as(f32, 0.95), lr }),
            .adam => {
                t.adam_t += 1;
                const tf: f32 = @floatFromInt(t.adam_t);
                const c1 = 1 - std.math.pow(f32, 0.9, tf);
                const c2 = 1 - std.math.pow(f32, 0.999, tf);
                try g.launch(k.adam, .{ blocks, 1, 1 }, 256, .{ t.params, t.grads, t.opt1, t.opt2, n, lr, c1, c2 });
            },
        }
        try g.memset32(t.grads, 0, n);
    }
};

/// Bump allocator over byte offsets (sizes given in 4-byte elements, 256-byte aligned).
const Carver = struct {
    off: usize = 0,
    fn take(c: *Carver, elems: usize) usize {
        return c.takeBytes(@max(elems, 1) * 4);
    }
    fn takeBytes(c: *Carver, bytes: usize) usize {
        const p = c.off;
        c.off += std.mem.alignForward(usize, bytes, 256);
        return p;
    }
};

/// Lays out one layer's activations starting at `c.off`, relative to `base`.
fn carveLayer(c: *Carver, base: DevPtr, L: usize, s: usize, H: usize, A: usize, r: usize) Acts {
    var act: Acts = undefined;
    inline for (.{ "h1", "q", "k", "v" }) |f| @field(act, f) = base + c.take(L * s);
    act.p = base + c.take(H * L * L);
    inline for (.{ "a", "x1", "h2", "qc" }) |f| @field(act, f) = base + c.take(L * s);
    act.pc = base + c.take(H * L * A);
    inline for (.{ "ac", "x2", "h3" }) |f| @field(act, f) = base + c.take(L * s);
    act.u = base + c.take(L * 4 * s);
    act.g = base + c.take(L * 4 * s);
    for (0..3) |j| {
        act.mean[j] = base + c.take(L);
        act.rstd[j] = base + c.take(L);
    }
    for (&act.z) |*z| z.* = base + c.take(L * r);
    return act;
}
