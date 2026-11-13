//! Runs the transformer on an NVIDIA GPU.
//!
//! Data movement is the part that decides speed, so it is kept to a minimum:
//!
//! * Weights are uploaded **once**, as f16 straight from the model file,
//!   into a single device allocation. The upload streams through two
//!   page-locked ("pinned") staging buffers, so the CPU fills one chunk while
//!   the GPU's DMA engine copies the other.
//! * Per window, only the raw audio samples go up (64 KB per second of audio,
//!   via pinned memory). Padding, STFT, log-mel, convolutions, encoder,
//!   cross-attention keys/values and the KV cache are all computed on the
//!   GPU and never leave it.
//! * Per generated token, the GPU picks the token itself and sends back 12
//!   bytes (token id + two log-probs) instead of the 200 KB logit vector.
//! * A personal adapter is merged into the weights on the GPU (`setAdapter`).
const std = @import("std");
const cuda = @import("cuda.zig");
const ptx = @import("stt_ptx");
const model_mod = @import("../model.zig");
const Model = model_mod.Model;
const Linear = model_mod.Linear;
const adapter = @import("../adapter.zig");
const mel = @import("../mel.zig");
const Allocator = std.mem.Allocator;
pub const DevPtr = cuda.DevPtr;

const shared = @import("shared.zig");
pub const flag_bias = shared.flag_bias;
pub const flag_gelu = shared.flag_gelu;
pub const flag_accum = shared.flag_accum;
pub const head_dim = shared.head_dim;
const split = shared.split;
const part_stride = shared.part_stride;
const n_bins = shared.n_bins;

comptime {
    if (shared.n_fft != mel.n_fft) @compileError("GPU and host STFT sizes differ");
}

pub const available = ptx.enabled and cuda.supported;

pub const DLinear = struct { w: DevPtr, b: DevPtr, n_in: u32, n_out: u32 };
pub const DNorm = struct { w: DevPtr, b: DevPtr };
pub const DAttn = struct { q: DLinear, k: DLinear, v: DLinear, o: DLinear };
pub const DEncLayer = struct { attn_ln: DNorm, attn: DAttn, mlp_ln: DNorm, fc1: DLinear, fc2: DLinear };
pub const DDecLayer = struct { attn_ln: DNorm, attn: DAttn, cross_ln: DNorm, cross: DAttn, mlp_ln: DNorm, fc1: DLinear, fc2: DLinear };

pub const Kernels = struct {
    gemm: cuda.Function,
    gemm_f32: cuda.Function,
    gemv: cuda.Function,
    pad_audio: cuda.Function,
    mel_power: cuda.Function,
    reduce_max: cuda.Function,
    mel_norm: cuda.Function,
    layernorm: cuda.Function,
    attn_full: cuda.Function,
    attn_one_part: cuda.Function,
    attn_one_combine: cuda.Function,
    im2col1: cuda.Function,
    im2col2: cuda.Function,
    embed: cuda.Function,
    pick: cuda.Function,
    lora_merge: cuda.Function,
    // training
    embed_seq: cuda.Function,
    layernorm_fwd: cuda.Function,
    layernorm_back: cuda.Function,
    softmax_rows: cuda.Function,
    softmax_back: cuda.Function,
    gelu_fwd: cuda.Function,
    gelu_back: cuda.Function,
    xent: cuda.Function,
    sumsq: cuda.Function,
    clip_scale: cuda.Function,
    adam: cuda.Function,
    muon: cuda.Function,
};

/// Device scratch buffers for inference (f32 unless noted).
const Bufs = struct {
    audio: DevPtr,
    padded: DevPtr,
    stft: DevPtr,
    scal: DevPtr,
    spec: DevPtr,
    col1: DevPtr,
    h1: DevPtr,
    col2: DevPtr,
    x: DevPtr,
    h: DevPtr,
    q: DevPtr,
    k: DevPtr,
    v: DevPtr,
    att: DevPtr,
    mlp: DevPtr,
    cross_k: DevPtr,
    cross_v: DevPtr,
    self_k: DevPtr,
    self_v: DevPtr,
    dx: DevPtr,
    dh: DevPtr,
    dq: DevPtr,
    datt: DevPtr,
    dmlp: DevPtr,
    logits: DevPtr,
    part: DevPtr,
    res: DevPtr, // u32[4]
};

pub const Pick = struct { token: u32, logprob: f32, nosp_logprob: f32 };

/// GPU time spent in the last window, from CUDA events (milliseconds).
pub const Timing = struct {
    encode_ms: f32 = 0,
    cross_ms: f32 = 0,
    decode_ms: f32 = 0,
};

/// One matrix operand of `Engine.mm`: element (i, p) is at
/// ptr[batch*b + i*i_stride + p*p_stride] (strides in elements).
pub const Operand = struct { ptr: DevPtr, i: u32, p: u32, b: u32 = 0 };

pub const Engine = struct {
    gpu: cuda.Gpu,
    module: cuda.Module,
    k: Kernels,
    model: *const Model,
    hp: model_mod.HParams,
    weights: DevPtr,
    weight_bytes: usize,
    scratch: DevPtr,
    scratch_bytes: usize,
    b: Bufs,
    /// Pinned host memory: audio upload area + 16-byte pick result.
    pinned: []align(4096) u8,
    // Device copies of the model.
    dft: DevPtr, // f32 [402][400] windowed DFT basis
    filters: DevPtr, // f32 [n_mels][201]
    conv1: DLinear,
    conv2: DLinear,
    enc_pos: DevPtr,
    enc: []DEncLayer,
    enc_ln_post: DNorm,
    tok_emb: DevPtr, // f16
    dec_pos: DevPtr,
    dec: []DDecLayer,
    dec_ln: DNorm,
    ev: [4]cuda.Event,
    /// Recorded after each audio upload; the pinned buffer is reused only once
    /// it has fired (back-to-back `encode` calls, e.g. in training).
    up_fence: cuda.Event,
    upload_ms: f32 = 0,
    /// Version tag of the adapter merged into the weights (0 = none).
    adapter_version: u32 = 0,

    /// Opens the first NVIDIA GPU and uploads `model`'s weights. `model`
    /// must outlive the engine (it is re-read by `setAdapter`). Fails with
    /// error.CudaUnavailable when there is no NVIDIA driver/GPU.
    pub fn init(gpa: Allocator, model: *const Model) !*Engine {
        if (!available) return error.CudaUnavailable;
        const hp = model.hp;
        if (hp.n_audio_state / hp.n_audio_head != head_dim or hp.n_text_state / hp.n_text_head != head_dim)
            return error.UnsupportedHeadDim;
        if (hp.n_audio_state % 8 != 0 or hp.n_text_state % 8 != 0) return error.UnsupportedModelShape;
        if (model.mel_filters.len != hp.n_mels * n_bins) return error.BadFilterbank;
        if (hp.n_audio_ctx * 2 != mel.n_frames) return error.UnsupportedAudioContext;

        var gpu = try cuda.Gpu.open(0);
        errdefer gpu.close();

        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = undefined;
        e.gpu = gpu;
        e.model = model;
        e.hp = hp;
        e.adapter_version = 0;
        e.weights = 0;
        e.scratch = 0;
        e.pinned = &.{};
        e.enc = try gpa.alloc(DEncLayer, hp.n_audio_layer);
        errdefer gpa.free(e.enc);
        e.dec = try gpa.alloc(DDecLayer, hp.n_text_layer);
        errdefer gpa.free(e.dec);

        e.module = try gpu.loadModule(ptx.source);
        errdefer gpu.unloadModule(e.module);
        inline for (@typeInfo(Kernels).@"struct".field_names) |name| {
            @field(e.k, name) = try gpu.function(e.module, name);
        }
        e.ev = @splat(null);
        errdefer for (e.ev) |ev| if (ev != null) gpu.destroyEvent(ev);
        for (&e.ev) |*ev| ev.* = try gpu.event();
        e.up_fence = try gpu.fence();
        errdefer gpu.destroyEvent(e.up_fence);

        // Weights: plan sizes, allocate once, then stream them up.
        var plan: Uploader = .{ .gpu = &e.gpu };
        try e.walkWeights(&plan, model);
        e.weight_bytes = plan.off;
        e.weights = try gpu.alloc(plan.off);
        errdefer gpu.free(e.weights);
        {
            var up: Uploader = .{ .gpu = &e.gpu, .base = e.weights };
            defer up.end(); // frees whatever begin() managed to allocate
            try up.begin();
            try gpu.record(e.ev[0]);
            try e.walkWeights(&up, model);
            try gpu.record(e.ev[1]);
            try gpu.sync();
            e.upload_ms = gpu.elapsed(e.ev[0], e.ev[1]);
        }

        try e.allocScratch();
        errdefer gpu.free(e.scratch);
        e.pinned = try gpu.allocPinned(std.mem.alignForward(usize, mel.n_samples * 4 + 16, 4096));
        return e;
    }

    pub fn deinit(e: *Engine, gpa: Allocator) void {
        const g = &e.gpu;
        g.makeCurrent();
        g.sync() catch {};
        for (e.ev) |ev| g.destroyEvent(ev);
        g.destroyEvent(e.up_fence);
        g.freePinned(e.pinned);
        g.free(e.scratch);
        g.free(e.weights);
        g.unloadModule(e.module);
        gpa.free(e.enc);
        gpa.free(e.dec);
        g.close();
        gpa.destroy(e);
    }

    pub fn deviceName(e: *const Engine) []const u8 {
        return e.gpu.deviceName();
    }

    fn walkWeights(e: *Engine, up: *Uploader, m: *const Model) !void {
        const basis = mel.dftBasis();
        e.dft = try up.put(f32, @as([*]const f32, @ptrCast(basis))[0 .. @sizeOf(mel.Basis) / 4]);
        e.filters = try up.put(f32, m.mel_filters);
        e.conv1 = try up.linear(m.conv1);
        e.conv2 = try up.linear(m.conv2);
        e.enc_pos = try up.put(f32, m.enc_pos);
        for (e.enc, m.enc_layers) |*d, l| d.* = .{
            .attn_ln = try up.norm(l.attn_ln),
            .attn = try up.attn(l.attn),
            .mlp_ln = try up.norm(l.mlp_ln),
            .fc1 = try up.linear(l.fc1),
            .fc2 = try up.linear(l.fc2),
        };
        e.enc_ln_post = try up.norm(m.enc_ln_post);
        e.tok_emb = try up.put(f16, m.tok_emb);
        e.dec_pos = try up.put(f32, m.dec_pos);
        for (e.dec, m.dec_layers) |*d, l| d.* = .{
            .attn_ln = try up.norm(l.attn_ln),
            .attn = try up.attn(l.attn),
            .cross_ln = try up.norm(l.cross_ln),
            .cross = try up.attn(l.cross),
            .mlp_ln = try up.norm(l.mlp_ln),
            .fc1 = try up.linear(l.fc1),
            .fc2 = try up.linear(l.fc2),
        };
        e.dec_ln = try up.norm(m.dec_ln);
    }

    /// The device copy of a decoder layer's adapted linear layer.
    pub fn slot(e: *const Engine, layer: usize, s: adapter.Slot) DLinear {
        const l = e.dec[layer];
        return switch (s) {
            .q => l.attn.q,
            .k => l.attn.k,
            .v => l.attn.v,
            .o => l.attn.o,
            .cross_q => l.cross.q,
            .cross_o => l.cross.o,
            .fc1 => l.fc1,
            .fc2 => l.fc2,
        };
    }

    /// Restores the base decoder weights and, if `ad` is given, folds it in
    /// (W += scale·B·A) on the GPU. `version` is just recorded for callers.
    pub fn setAdapter(e: *Engine, ad: ?*const adapter.Adapter, version: u32) !void {
        const g = &e.gpu;
        g.makeCurrent();
        {
            var up: Uploader = .{ .gpu = g };
            defer up.end();
            try up.begin();
            for (0..e.hp.n_text_layer) |l| for (0..adapter.n_slots) |si| {
                const s: adapter.Slot = @enumFromInt(si);
                try up.putAt(e.slot(l, s).w, f16, adapter.slotLinear(e.model, l, s).w);
            };
            try g.sync();
        }
        e.adapter_version = 0;
        const a = ad orelse return;
        if (a.n_layers != e.hp.n_text_layer or a.n_state != e.hp.n_text_state) return error.AdapterModelMismatch;
        const params = try g.alloc(a.buf.len * 4);
        defer g.free(params);
        {
            var up: Uploader = .{ .gpu = g };
            defer up.end();
            try up.begin();
            try up.putAt(params, f32, a.buf);
            for (0..a.n_layers) |l| for (0..adapter.n_slots) |si| {
                const s: adapter.Slot = @enumFromInt(si);
                const d = e.slot(l, s);
                const off = a.offsetOf(l, s);
                const lo = a.get(l, s);
                try g.launch(e.k.lora_merge, .{ (d.n_out * d.n_in + 255) / 256, 1, 1 }, 256, .{
                    d.w, params + off * 4, params + (off + lo.a.len) * 4, d.n_out, d.n_in, @as(u32, @intCast(a.rank)), a.scale(),
                });
            };
            try g.sync();
        }
        e.adapter_version = version;
    }

    fn allocScratch(e: *Engine) !void {
        const hp = e.hp;
        const T = mel.n_frames;
        const A = hp.n_audio_ctx;
        const ns = hp.n_audio_state;
        const ts = hp.n_text_state;
        const max_kv = @max(A, hp.n_text_ctx);
        const sizes = [_]usize{
            mel.n_samples, // audio
            mel.n_samples + mel.n_fft, // padded
            T * 2 * n_bins, // stft
            4, // scal
            hp.n_mels * T, // spec
            T * hp.n_mels * 3, // col1
            T * ns, // h1
            A * ns * 3, // col2
            A * ns, A * ns, A * ns, A * ns, A * ns, A * ns, // x h q k v att
            A * ns * 4, // mlp
            hp.n_text_layer * A * ts, hp.n_text_layer * A * ts, // cross k/v
            hp.n_text_layer * hp.n_text_ctx * ts, hp.n_text_layer * hp.n_text_ctx * ts, // self k/v
            ts, ts, ts, ts, ts * 4, // dx dh dq datt dmlp
            hp.n_vocab, // logits
            hp.n_text_head * ((max_kv + split - 1) / split) * part_stride, // part
            4, // res
        };
        var total: usize = 0;
        for (sizes) |n| total += std.mem.alignForward(usize, n * 4, 256);
        e.scratch = try e.gpu.alloc(total);
        e.scratch_bytes = total;
        var off: usize = 0;
        inline for (@typeInfo(Bufs).@"struct".field_names, 0..) |name, i| {
            @field(e.b, name) = e.scratch + off;
            off += std.mem.alignForward(usize, sizes[i] * 4, 256);
        }
    }

    // ------------------------------------------------------------ launches

    /// out(i, j) [+]= alpha · Σ_p A(i, p) · W(p, j) (+ bias, gelu), batched
    /// over `batch`. `w16` selects f16 (true) or f32 weights. See kernels.gemm.
    pub fn mm(e: *Engine, comptime w16: bool, out: Operand, a: Operand, w: Operand, bias: DevPtr, m: u32, n: u32, k: u32, flags: u32, alpha: f32, batch: u32) !void {
        const f = if (w16) e.k.gemm else e.k.gemm_f32;
        try e.gpu.launch(f, .{ (n + 63) / 64, (m + 63) / 64, batch }, 256, .{
            out.ptr, out.i,  out.b,
            a.ptr,   a.i,    a.p,
            a.b,     w.ptr,  w.p,
            w.i,     w.b,    bias,
            m,       n,      k,
            flags,   alpha,
        });
    }

    /// out[m][n_out] (op)= in[m][n_in] · Wᵀ + b. Uses gemv for single rows.
    pub fn linear(e: *Engine, out: DevPtr, in: DevPtr, m: u32, l: DLinear, extra: u32) !void {
        const flags = extra | (if (l.b != 0) flag_bias else 0);
        if (m == 1) {
            try e.gpu.launch(e.k.gemv, .{ (l.n_out + 7) / 8, 1, 1 }, 256, .{ out, in, l.w, l.b, l.n_out, l.n_in, flags });
        } else {
            try e.mm(true, .{ .ptr = out, .i = l.n_out, .p = 1 }, .{ .ptr = in, .i = l.n_in, .p = 1 }, .{ .ptr = l.w, .i = l.n_in, .p = 1 }, l.b, m, l.n_out, l.n_in, flags, 1, 1);
        }
    }

    fn layerNorm(e: *Engine, out: DevPtr, x: DevPtr, rows: u32, n: u32, nm: DNorm) !void {
        try e.gpu.launch(e.k.layernorm, .{ rows, 1, 1 }, 256, .{ out, x, nm.w, nm.b, n });
    }

    fn attendOne(e: *Engine, out: DevPtr, q: DevPtr, k: DevPtr, v: DevPtr, n_kv: u32) !void {
        const s: u32 = @intCast(e.hp.n_text_state);
        const heads: u32 = @intCast(e.hp.n_text_head);
        const nsplit = (n_kv + split - 1) / split;
        const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
        try e.gpu.launch(e.k.attn_one_part, .{ nsplit, heads, 1 }, 128, .{ e.b.part, q, k, v, n_kv, s, scale });
        try e.gpu.launch(e.k.attn_one_combine, .{ heads, 1, 1 }, 64, .{ out, e.b.part, nsplit });
    }

    /// Uploads up to 30 s of 16 kHz audio, computes its log-mel spectrogram,
    /// runs the audio encoder and precomputes every decoder layer's
    /// cross-attention keys/values. All results stay on the GPU.
    pub fn encode(e: *Engine, samples_in: []const f32) !void {
        const g = &e.gpu;
        g.makeCurrent();
        const hp = e.hp;
        const T: u32 = mel.n_frames;
        const A: u32 = @intCast(hp.n_audio_ctx);
        const ns: u32 = @intCast(hp.n_audio_state);
        const nm: u32 = @intCast(hp.n_mels);
        const samples = samples_in[0..@min(samples_in.len, mel.n_samples)];
        const n: u32 = @intCast(samples.len);
        const total: u32 = mel.n_samples;

        try g.wait(e.up_fence); // previous upload out of the pinned buffer is done
        try g.record(e.ev[0]);
        const bytes = std.mem.sliceAsBytes(samples);
        @memcpy(e.pinned[0..bytes.len], bytes);
        if (n > 0) try g.upload(e.b.audio, e.pinned[0..bytes.len]);
        try g.record(e.up_fence);

        // Log-mel spectrogram: pad, STFT as one matmul (frames overlap, so
        // row t of the "matrix" starts at sample t*hop), power, mel, log.
        try g.launch(e.k.pad_audio, .{ (total + mel.n_fft + 255) / 256, 1, 1 }, 256, .{ e.b.padded, e.b.audio, n, total });
        try e.mm(false, .{ .ptr = e.b.stft, .i = 2 * n_bins, .p = 1 }, .{ .ptr = e.b.padded, .i = mel.hop, .p = 1 }, .{ .ptr = e.dft, .i = mel.n_fft, .p = 1 }, 0, T, 2 * n_bins, mel.n_fft, 0, 1, 1);
        try g.launch(e.k.mel_power, .{ T, 1, 1 }, 128, .{ e.b.spec, e.b.stft, e.filters, nm, T });
        try g.launch(e.k.reduce_max, .{ 1, 1, 1 }, 1024, .{ e.b.scal, e.b.spec, nm * T });
        try g.launch(e.k.mel_norm, .{ (nm * T + 255) / 256, 1, 1 }, 256, .{ e.b.spec, nm * T, e.b.scal });

        try g.launch(e.k.im2col1, .{ (T * nm * 3 + 255) / 256, 1, 1 }, 256, .{ e.b.col1, e.b.spec, T, nm });
        try e.linear(e.b.h1, e.b.col1, T, e.conv1, flag_gelu);
        try g.launch(e.k.im2col2, .{ (A * ns * 3 + 255) / 256, 1, 1 }, 256, .{ e.b.col2, e.b.h1, A, T, ns });
        try g.copy(e.b.x, e.enc_pos, A * ns * 4);
        try e.linear(e.b.x, e.b.col2, A, e.conv2, flag_gelu | flag_accum); // x = pos + gelu(conv2)

        const heads: u32 = @intCast(hp.n_audio_head);
        const scale: f32 = 1.0 / @sqrt(@as(f32, head_dim));
        const b = e.b;
        for (e.enc) |l| {
            try e.layerNorm(b.h, b.x, A, ns, l.attn_ln);
            try e.linear(b.q, b.h, A, l.attn.q, 0);
            try e.linear(b.k, b.h, A, l.attn.k, 0);
            try e.linear(b.v, b.h, A, l.attn.v, 0);
            try g.launch(e.k.attn_full, .{ (A + 63) / 64, heads, 1 }, 64, .{ b.att, b.q, b.k, b.v, A, ns, scale });
            try e.linear(b.x, b.att, A, l.attn.o, flag_accum);
            try e.layerNorm(b.h, b.x, A, ns, l.mlp_ln);
            try e.linear(b.mlp, b.h, A, l.fc1, flag_gelu);
            try e.linear(b.x, b.mlp, A, l.fc2, flag_accum);
        }
        try e.layerNorm(b.x, b.x, A, ns, e.enc_ln_post);
        try g.record(e.ev[1]);

        const ts: u32 = @intCast(hp.n_text_state);
        for (e.dec, 0..) |l, i| {
            const off = i * A * ts * 4;
            try e.linear(b.cross_k + off, b.x, A, l.cross.k, 0);
            try e.linear(b.cross_v + off, b.x, A, l.cross.v, 0);
        }
        try g.record(e.ev[2]);
    }

    /// Cross-attention keys/values of decoder layer `layer` from the last `encode`.
    pub fn crossKV(e: *const Engine, layer: usize) struct { k: DevPtr, v: DevPtr } {
        const off = layer * e.hp.n_audio_ctx * e.hp.n_text_state * 4;
        return .{ .k = e.b.cross_k + off, .v = e.b.cross_v + off };
    }

    /// Feeds one token at position `pos` through the decoder (KV-cached).
    /// If `want_logits`, also computes the vocabulary logits (read via `pick`).
    pub fn step(e: *Engine, token: u32, pos: u32, want_logits: bool) !void {
        const g = &e.gpu;
        const hp = e.hp;
        const s: u32 = @intCast(hp.n_text_state);
        const A: u32 = @intCast(hp.n_audio_ctx);
        const tctx = hp.n_text_ctx;
        const b = e.b;
        try g.launch(e.k.embed, .{ (s + 255) / 256, 1, 1 }, 256, .{ b.dx, e.tok_emb, e.dec_pos, token, pos, s });
        for (e.dec, 0..) |l, i| {
            const sk = b.self_k + i * tctx * s * 4;
            const sv = b.self_v + i * tctx * s * 4;
            try e.layerNorm(b.dh, b.dx, 1, s, l.attn_ln);
            try e.linear(b.dq, b.dh, 1, l.attn.q, 0);
            try e.linear(sk + pos * s * 4, b.dh, 1, l.attn.k, 0);
            try e.linear(sv + pos * s * 4, b.dh, 1, l.attn.v, 0);
            try e.attendOne(b.datt, b.dq, sk, sv, pos + 1);
            try e.linear(b.dx, b.datt, 1, l.attn.o, flag_accum);

            try e.layerNorm(b.dh, b.dx, 1, s, l.cross_ln);
            try e.linear(b.dq, b.dh, 1, l.cross.q, 0);
            const off = i * A * s * 4;
            try e.attendOne(b.datt, b.dq, b.cross_k + off, b.cross_v + off, A);
            try e.linear(b.dx, b.datt, 1, l.cross.o, flag_accum);

            try e.layerNorm(b.dh, b.dx, 1, s, l.mlp_ln);
            try e.linear(b.dmlp, b.dh, 1, l.fc1, flag_gelu);
            try e.linear(b.dx, b.dmlp, 1, l.fc2, flag_accum);
        }
        if (!want_logits) return;
        try e.layerNorm(b.dh, b.dx, 1, s, e.dec_ln);
        // Output projection is tied to the token embedding.
        try e.linear(b.logits, b.dh, 1, .{ .w = e.tok_emb, .b = 0, .n_in = s, .n_out = @intCast(hp.n_vocab) }, 0);
    }

    /// Picks the most likely token from the last logits, excluding ids >=
    /// `ban_from` and `ban` (use `none`). Waits for the GPU.
    pub fn pick(e: *Engine, ban_from: u32, ban: [2]u32, nosp: u32) !Pick {
        const g = &e.gpu;
        try g.launch(e.k.pick, .{ 1, 1, 1 }, 1024, .{ e.b.res, e.b.logits, @as(u32, @intCast(e.hp.n_vocab)), ban_from, ban[0], ban[1], nosp });
        const res = e.pinned[e.pinned.len - 16 ..];
        try g.download(res[0..12], e.b.res);
        try g.record(e.ev[3]);
        try g.wait(e.ev[3]);
        const w = std.mem.bytesAsSlice(u32, res[0..12]);
        if (w[0] >= e.hp.n_vocab) return error.GpuBadResult;
        return .{ .token = w[0], .logprob = @bitCast(w[1]), .nosp_logprob = @bitCast(w[2]) };
    }

    /// GPU time of the last window's stages (from CUDA events).
    pub fn timings(e: *const Engine) Timing {
        const g = &e.gpu;
        return .{
            .encode_ms = g.elapsed(e.ev[0], e.ev[1]),
            .cross_ms = g.elapsed(e.ev[1], e.ev[2]),
            .decode_ms = g.elapsed(e.ev[2], e.ev[3]),
        };
    }

    pub const none: u32 = 0xffff_ffff;
};

/// Bump-allocates weights inside one device buffer and streams them up through
/// two pinned staging buffers (fill chunk N+1 while chunk N is in flight).
/// With `base == 0` it only measures the space needed.
pub const Uploader = struct {
    gpu: *const cuda.Gpu,
    base: DevPtr = 0,
    off: usize = 0,
    stage: [2][]align(4096) u8 = .{ &.{}, &.{} },
    fence: [2]cuda.Event = .{ null, null },
    used: [2]bool = .{ false, false },
    cur: u1 = 0,

    const stage_bytes = 4 << 20;

    pub fn begin(u: *Uploader) !void {
        for (0..2) |i| {
            u.stage[i] = try u.gpu.allocPinned(stage_bytes);
            u.fence[i] = try u.gpu.fence();
        }
    }
    pub fn end(u: *Uploader) void {
        u.gpu.sync() catch {};
        for (0..2) |i| {
            if (u.stage[i].len > 0) u.gpu.freePinned(u.stage[i]);
            if (u.fence[i] != null) u.gpu.destroyEvent(u.fence[i]);
        }
    }

    fn reserve(u: *Uploader, bytes: usize) DevPtr {
        const p = u.base + u.off;
        u.off += std.mem.alignForward(usize, bytes, 256);
        return p;
    }

    fn put(u: *Uploader, comptime T: type, src: []const T) !DevPtr {
        const dst = u.reserve(src.len * @sizeOf(T));
        if (u.base == 0) return 0;
        try u.putAt(dst, T, src);
        return dst;
    }

    /// Copies `src` to device address `dst`, chunked through the staging buffers.
    pub fn putAt(u: *Uploader, dst: DevPtr, comptime T: type, src: []const T) !void {
        const bytes = std.mem.sliceAsBytes(src);
        var i: usize = 0;
        while (i < bytes.len) {
            const n: usize = @min(stage_bytes, bytes.len - i);
            const c = u.cur;
            // Wait until the DMA from this staging buffer has finished before refilling it.
            if (u.used[c]) try u.gpu.wait(u.fence[c]);
            @memcpy(u.stage[c][0..n], bytes[i..][0..n]);
            try u.gpu.upload(dst + i, u.stage[c][0..n]);
            try u.gpu.record(u.fence[c]);
            u.used[c] = true;
            u.cur ^= 1;
            i += n;
        }
    }

    fn linear(u: *Uploader, l: Linear) !DLinear {
        return .{
            .w = try u.put(f16, l.w),
            .b = if (l.b) |b| try u.put(f32, b) else 0,
            .n_in = @intCast(l.n_in),
            .n_out = @intCast(l.n_out),
        };
    }
    fn norm(u: *Uploader, n: model_mod.Norm) !DNorm {
        return .{ .w = try u.put(f32, n.w), .b = try u.put(f32, n.b) };
    }
    fn attn(u: *Uploader, a: model_mod.Attn) !DAttn {
        return .{ .q = try u.linear(a.q), .k = try u.linear(a.k), .v = try u.linear(a.v), .o = try u.linear(a.o) };
    }
};
