//! Runs the transformer through Core ML (iPhone / Mac: the Neural Engine,
//! GPU or CPU, as Core ML places each op), from two ML Programs generated
//! in Zig from the ggml weights (net.zig, TranslateCoreML's MIL writer).
//! No Python, no coremltools, nothing prebuilt to ship: the packages are
//! written into `cache_dir`, compiled by Core ML on the device and cached
//! (key: weights, adapter, graph source, opset, compute units).
//!
//! Same interface as gpu/engine.zig and npu/engine.zig, so transcriber.zig
//! drives it unchanged:
//!
//!   init(gpa, model, opts) / deinit(gpa)
//!   encode(samples)                  up to 30 s of 16 kHz mono f32 audio
//!   step(token, pos, want_logits)    one decoder token (KV-cached)
//!   pick(ban_from, ban, nosp) Pick   greedy choice over the last logits
//!   setAdapter(?*const Adapter, version)
//!   deviceName()
//!
//! On the host (single-threaded, no worker threads): the log-mel front end
//! (frontend.zig), the token + position embedding row, the attention mask
//! row, writing each step's new K/V row into the cache, and the greedy pick
//! over the fp16 logits. Everything else is in the graphs.
//!
//! Host buffers are fp16 and bound to the Core ML inputs without copies
//! (MLMultiArray over our memory); the encoder's cross K/V outputs are read
//! straight into the decoder's input buffers once per window.
//!
//! `runtime = .reference` runs the very same programs with TranslateCoreML's
//! host interpreter instead of Core ML (fp16 rounding emulated), on any OS:
//! that is how the graphs are checked against the GPU on the desktop.
const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const coreml = @import("coreml");
const mil = coreml.mil;
const rt = coreml.runtime;
const model_mod = @import("../model.zig");
const Model = model_mod.Model;
const adapter_mod = @import("../adapter.zig");
const Adapter = adapter_mod.Adapter;
const mel = @import("../mel.zig");
const gpu_engine = @import("../gpu/engine.zig");
const frontend = @import("frontend.zig");
pub const net = @import("net.zig");

const log = std.log.scoped(.coreml);

pub const Pick = gpu_engine.Pick;

/// True on Apple targets, where Core ML itself can run the graphs.
pub const has_coreml = coreml.has_runtime;

pub const Runtime = enum {
    /// Core ML (Apple targets only).
    coreml,
    /// TranslateCoreML's host interpreter (any OS; slow, for checking).
    reference,
};

/// MLComputeUnits (mirrored so the options exist on every OS).
pub const ComputeUnits = enum(isize) {
    cpu_only = 0,
    cpu_and_gpu = 1,
    all = 2,
    cpu_and_neural_engine = 3,
};

pub const Options = struct {
    runtime: Runtime = if (has_coreml) .coreml else .reference,
    /// Writable folder for the generated packages and compiled models
    /// (required with .coreml), e.g. the app's Library/Caches.
    cache_dir: ?[]const u8 = null,
    compute_units: ComputeUnits = .cpu_and_neural_engine,
    /// GELU form: the GPU's tanh approximation (default) or exact erf.
    gelu: mil.GeluMode = .tanh_approximation,
    /// MIL opset / specification version of the generated programs.
    target: mil.Target = .ios17,
    /// Reference runtime: round every fp16 value as Core ML would store it.
    emulate_fp16: bool = true,
};

/// Wall-clock time of the last calls (milliseconds).
pub const Timing = struct {
    /// Generating + compiling + loading each program (or loading it from the cache).
    encoder_prepare_ms: f64 = 0,
    decoder_prepare_ms: f64 = 0,
    encoder_from_cache: bool = false,
    decoder_from_cache: bool = false,
    /// Last encode(): host log-mel, and the encoder prediction.
    mel_ms: f64 = 0,
    encode_ms: f64 = 0,
    /// Last step() prediction, and the sum since encode().
    step_ms: f64 = 0,
    steps_ms: f64 = 0,
    steps: u32 = 0,
};

/// One input or output of a program, bound to an engine buffer.
const Binding = struct { name: []const u8, shape: []const usize, buf: []f16 };

/// A loaded program.
const Graph = struct {
    inputs: []Binding = &.{},
    outputs: []Binding = &.{},
    /// .reference: the program (kept) and its interpreter.
    builder: ?*mil.Builder = null,
    interp: ?coreml.ref.Interp = null,
    scratch: [][]f32 = &.{},
    /// Merged adapter weights the program borrows (reference runtime).
    owned_weights: ?[]f16 = null,
    /// .coreml: the model and one MLMultiArray over each input buffer.
    cm: if (has_coreml) ?CoreMLGraph else void = if (has_coreml) null else {},

    const CoreMLGraph = struct {
        model: rt.Model,
        arrays: []rt.MultiArray,
        feeds: []rt.Model.Input,
    };

    fn deinit(g: *Graph, gpa: Allocator) void {
        if (has_coreml) if (g.cm) |*c| {
            for (c.arrays) |*a| a.deinit();
            gpa.free(c.arrays);
            gpa.free(c.feeds);
            c.model.deinit();
        };
        if (g.interp) |*it| it.deinit();
        if (g.builder) |b| {
            b.deinit();
            gpa.destroy(b);
        }
        for (g.scratch) |s| gpa.free(s);
        gpa.free(g.scratch);
        if (g.owned_weights) |w| gpa.free(w);
        gpa.free(g.inputs);
        gpa.free(g.outputs);
        g.* = .{};
    }

    fn run(g: *Graph, gpa: Allocator) !void {
        if (has_coreml) if (g.cm) |*c| {
            const pool = coreml.objc.AutoreleasePool.push();
            defer pool.pop();
            var pred = c.model.predict(c.feeds) catch |e| {
                log.err("prediction failed: {s}", .{rt.lastError()});
                return e;
            };
            defer pred.deinit();
            for (g.outputs) |o| {
                var a = pred.array(o.name) orelse return error.CoreMLMissingOutput;
                defer a.deinit();
                if (a.count() != o.buf.len) return error.CoreMLBadOutput;
                switch (a.dtype) {
                    .float16 => a.readF16(o.buf),
                    else => {
                        const tmp = try gpa.alloc(f32, o.buf.len);
                        defer gpa.free(tmp);
                        a.readF32(tmp);
                        for (o.buf, tmp) |*d, v| d.* = @floatCast(v);
                    },
                }
            }
            return;
        };
        const it = &(g.interp orelse return error.NotLoaded);
        var named: [64]coreml.ref.Named = undefined;
        for (g.inputs, g.scratch, 0..) |in, s, i| {
            for (s, in.buf) |*d, v| d.* = @floatCast(v);
            named[i] = .{ .name = in.name, .shape = in.shape, .data = s };
        }
        try it.run(named[0..g.inputs.len]);
        for (g.outputs) |o| {
            const t = it.output(o.name) orelse return error.MissingOutput;
            for (o.buf, t.data) |*d, v| d.* = @floatCast(v);
        }
    }
};

pub const Engine = struct {
    /// Version tag of the adapter merged into the weights (0 = none).
    adapter_version: u32 = 0,

    gpa: Allocator,
    io: Io,
    model: *const Model,
    hp: model_mod.HParams,
    opts: Options,
    cache_dir: ?[]u8 = null,
    name: []const u8,
    model_key: u64 = 0,
    timing: Timing = .{},

    enc: Graph = .{},
    dec: Graph = .{},

    // Host buffers (fp16), bound to the programs' inputs/outputs.
    mel32: []f32,
    mel16: []f16,
    x: []f16,
    mask: []f16,
    cross: []f16, // [layer][k|v][heads][1500][64]: encoder outputs = decoder inputs
    self_kv: []f16, // [layer][k|v][heads][ctx][64]
    kv_new: []f16, // [layer][k|v][state]
    logits: []f16,
    have_logits: bool = false,

    pub fn init(gpa: Allocator, model: *const Model, opts: Options) !*Engine {
        const hp = model.hp;
        if (hp.n_audio_state != hp.n_audio_head * net.head_dim or hp.n_text_state != hp.n_text_head * net.head_dim)
            return error.UnsupportedHeadDim;
        if (hp.n_audio_ctx * 2 != mel.n_frames) return error.UnsupportedAudioContext;
        if (model.mel_filters.len != hp.n_mels * frontend.n_bins) return error.BadFilterbank;
        if (opts.runtime == .coreml) {
            if (!has_coreml) return error.CoreMLUnavailable;
            if (opts.cache_dir == null) return error.CoreMLNeedsCacheDir;
            coreml.objc.load() catch return error.CoreMLUnavailable;
        }
        const L = hp.n_text_layer;
        const A = hp.n_audio_ctx;
        const ts = hp.n_text_state;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.* = .{
            .gpa = gpa,
            .io = std.Io.Threaded.global_single_threaded.io(),
            .model = model,
            .hp = hp,
            .opts = opts,
            .name = switch (opts.runtime) {
                .coreml => switch (opts.compute_units) {
                    .cpu_and_neural_engine => "Core ML (Neural Engine)",
                    .all => "Core ML (all compute units)",
                    .cpu_and_gpu => "Core ML (GPU)",
                    .cpu_only => "Core ML (CPU)",
                },
                .reference => "Core ML graphs on the host (reference interpreter)",
            },
            .mel32 = &.{},
            .mel16 = &.{},
            .x = &.{},
            .mask = &.{},
            .cross = &.{},
            .self_kv = &.{},
            .kv_new = &.{},
            .logits = &.{},
        };
        errdefer e.freeAll();
        if (opts.cache_dir) |d| e.cache_dir = try gpa.dupe(u8, d);
        e.mel32 = try gpa.alloc(f32, hp.n_mels * mel.n_frames);
        e.mel16 = try gpa.alloc(f16, hp.n_mels * mel.n_frames);
        e.x = try gpa.alloc(f16, ts);
        e.mask = try gpa.alloc(f16, hp.n_text_ctx + 1);
        e.cross = try gpa.alloc(f16, L * 2 * A * ts);
        @memset(e.cross, 0);
        e.self_kv = try gpa.alloc(f16, L * 2 * hp.n_text_ctx * ts);
        @memset(e.self_kv, 0);
        e.kv_new = try gpa.alloc(f16, L * 2 * ts);
        e.logits = try gpa.alloc(f16, hp.n_vocab);

        if (opts.runtime == .coreml) e.model_key = modelHash(model);
        const t0 = e.now();
        try e.openEncoder();
        e.timing.encoder_prepare_ms = e.since(t0);
        try e.openDecoder(null, 0);
        return e;
    }

    pub fn deinit(e: *Engine, gpa: Allocator) void {
        _ = gpa; // the engine's own allocator (the same one) owns everything
        e.freeAll();
        e.gpa.destroy(e);
    }

    fn freeAll(e: *Engine) void {
        const gpa = e.gpa;
        e.enc.deinit(gpa);
        e.dec.deinit(gpa);
        gpa.free(e.mel32);
        gpa.free(e.mel16);
        gpa.free(e.x);
        gpa.free(e.mask);
        gpa.free(e.cross);
        gpa.free(e.self_kv);
        gpa.free(e.kv_new);
        gpa.free(e.logits);
        if (e.cache_dir) |d| gpa.free(d);
        e.cache_dir = null;
    }

    pub fn deviceName(e: *const Engine) []const u8 {
        return e.name;
    }

    // ------------------------------------------------------------ buffers

    fn crossSlice(e: *const Engine, layer: usize, which: usize) []f16 {
        const per = e.hp.n_audio_ctx * e.hp.n_text_state;
        return e.cross[(layer * 2 + which) * per ..][0..per];
    }
    fn selfSlice(e: *const Engine, layer: usize, which: usize) []f16 {
        const per = e.hp.n_text_ctx * e.hp.n_text_state;
        return e.self_kv[(layer * 2 + which) * per ..][0..per];
    }
    fn newSlice(e: *const Engine, layer: usize, which: usize) []f16 {
        const ts = e.hp.n_text_state;
        return e.kv_new[(layer * 2 + which) * ts ..][0..ts];
    }

    fn encoderBindings(e: *Engine) !struct { in: []Binding, out: []Binding } {
        const gpa = e.gpa;
        const hp = e.hp;
        const L = hp.n_text_layer;
        const in = try gpa.alloc(Binding, 1);
        errdefer gpa.free(in);
        in[0] = .{ .name = "mel", .shape = try e.shapeOf(&.{ 1, hp.n_mels, mel.n_frames }), .buf = e.mel16 };
        const out = try gpa.alloc(Binding, 2 * L);
        for (0..L) |l| for (0..2) |w| {
            out[l * 2 + w] = .{ .name = names.get(if (w == 0) .ck else .cv, l), .shape = try e.shapeOf(&.{ 1, hp.n_text_head, hp.n_audio_ctx, net.head_dim }), .buf = e.crossSlice(l, w) };
        };
        return .{ .in = in, .out = out };
    }

    fn decoderBindings(e: *Engine) !struct { in: []Binding, out: []Binding } {
        const gpa = e.gpa;
        const hp = e.hp;
        const L = hp.n_text_layer;
        const H = hp.n_text_head;
        const in = try gpa.alloc(Binding, 2 + 4 * L);
        errdefer gpa.free(in);
        in[0] = .{ .name = "x", .shape = try e.shapeOf(&.{ 1, hp.n_text_state }), .buf = e.x };
        in[1] = .{ .name = "mask", .shape = try e.shapeOf(&.{ 1, 1, 1, hp.n_text_ctx + 1 }), .buf = e.mask };
        for (0..L) |l| {
            const self_shape = try e.shapeOf(&.{ 1, H, hp.n_text_ctx, net.head_dim });
            const cross_shape = try e.shapeOf(&.{ 1, H, hp.n_audio_ctx, net.head_dim });
            in[2 + l * 4 + 0] = .{ .name = names.get(.sk, l), .shape = self_shape, .buf = e.selfSlice(l, 0) };
            in[2 + l * 4 + 1] = .{ .name = names.get(.sv, l), .shape = self_shape, .buf = e.selfSlice(l, 1) };
            in[2 + l * 4 + 2] = .{ .name = names.get(.ck, l), .shape = cross_shape, .buf = e.crossSlice(l, 0) };
            in[2 + l * 4 + 3] = .{ .name = names.get(.cv, l), .shape = cross_shape, .buf = e.crossSlice(l, 1) };
        }
        const out = try gpa.alloc(Binding, 2 * L + 1);
        for (0..L) |l| for (0..2) |w| {
            out[l * 2 + w] = .{ .name = names.get(if (w == 0) .kn else .vn, l), .shape = try e.shapeOf(&.{ 1, hp.n_text_state }), .buf = e.newSlice(l, w) };
        };
        out[2 * L] = .{ .name = "logits", .shape = try e.shapeOf(&.{ 1, hp.n_vocab }), .buf = e.logits };
        return .{ .in = in, .out = out };
    }

    /// Shapes live as long as the process (a few small arrays per engine).
    fn shapeOf(e: *Engine, s: []const usize) ![]const usize {
        _ = e;
        return shapes.intern(s);
    }

    // ------------------------------------------------------------ programs

    fn openEncoder(e: *Engine) !void {
        const b = try e.gpa.create(mil.Builder);
        b.* = try mil.Builder.init(e.gpa, .{ .target = e.opts.target, .description = "stt encoder" });
        var keep_builder = false;
        defer if (!keep_builder) {
            b.deinit();
            e.gpa.destroy(b);
        };
        try net.encoder(b, e.model, .{ .gelu = e.opts.gelu, .target = e.opts.target });
        const bind = try e.encoderBindings();
        var g: Graph = .{ .inputs = bind.in, .outputs = bind.out };
        errdefer g.deinit(e.gpa);
        const key = e.graphKey("enc", null);
        try e.load(&g, b, "stt-enc", key, &keep_builder, &e.timing.encoder_from_cache);
        e.enc.deinit(e.gpa);
        e.enc = g;
    }

    fn openDecoder(e: *Engine, ad: ?*const Adapter, version: u32) !void {
        const gpa = e.gpa;
        var merged: ?[]f16 = null;
        errdefer if (merged) |w| gpa.free(w);
        const slots = if (ad) |a| blk: {
            const r = try net.mergeAdapter(gpa, e.model, a);
            merged = r.buf;
            break :blk r.slots;
        } else try net.baseWeights(gpa, e.model);
        defer gpa.free(slots);

        const t0 = e.now();
        const b = try gpa.create(mil.Builder);
        b.* = try mil.Builder.init(gpa, .{ .target = e.opts.target, .description = "stt decoder step" });
        var keep_builder = false;
        defer if (!keep_builder) {
            b.deinit();
            gpa.destroy(b);
        };
        try net.decoder(b, e.model, .{ .gelu = e.opts.gelu, .target = e.opts.target }, .{ .w = slots });
        const bind = try e.decoderBindings();
        var g: Graph = .{ .inputs = bind.in, .outputs = bind.out };
        errdefer g.deinit(gpa);
        const key = e.graphKey("dec", ad);
        try e.load(&g, b, "stt-dec", key, &keep_builder, &e.timing.decoder_from_cache);
        // The reference program borrows the merged weights.
        if (keep_builder) {
            g.owned_weights = merged;
            merged = null;
        } else if (merged) |w| {
            gpa.free(w);
            merged = null;
        }
        e.timing.decoder_prepare_ms = e.since(t0);
        e.dec.deinit(gpa);
        e.dec = g;
        e.have_logits = false;
        e.adapter_version = if (ad != null) version else 0;
    }

    /// Readies `g` from builder `b`: the reference interpreter keeps the
    /// builder; Core ML writes the package (unless a compiled model for
    /// `key` is cached), compiles it and loads it.
    fn load(e: *Engine, g: *Graph, b: *mil.Builder, prefix: []const u8, key: u64, keep_builder: *bool, from_cache: *bool) !void {
        const gpa = e.gpa;
        from_cache.* = false;
        switch (e.opts.runtime) {
            .reference => {
                g.builder = b;
                keep_builder.* = true;
                g.interp = try coreml.ref.Interp.init(gpa, b);
                g.interp.?.emulate_fp16 = e.opts.emulate_fp16;
                g.scratch = try gpa.alloc([]f32, g.inputs.len);
                for (g.scratch) |*s| s.* = &.{};
                for (g.scratch, g.inputs) |*s, in| s.* = try gpa.alloc(f32, in.buf.len);
            },
            .coreml => if (has_coreml) {
                const dir = e.cache_dir.?;
                const name = try std.fmt.allocPrint(gpa, "{s}-{x:0>16}", .{ prefix, key });
                defer gpa.free(name);
                const compiled = try std.fmt.allocPrint(gpa, "{s}/{s}.mlmodelc", .{ dir, name });
                defer gpa.free(compiled);
                const cwd = Io.Dir.cwd();
                if (cwd.access(e.io, compiled, .{})) |_| {
                    from_cache.* = true;
                } else |_| {
                    try cwd.createDirPath(e.io, dir);
                    var d = try cwd.openDir(e.io, dir, .{});
                    defer d.close(e.io);
                    const pkg = try std.fmt.allocPrint(gpa, "{s}.mlpackage", .{name});
                    defer gpa.free(pkg);
                    try coreml.package.write(gpa, e.io, d, pkg, b);
                    const pkg_path = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir, pkg });
                    defer gpa.free(pkg_path);
                    const out = rt.compile(gpa, pkg_path, dir, name) catch |err| {
                        log.err("compiling {s}: {s}", .{ pkg, rt.lastError() });
                        return err;
                    };
                    gpa.free(out);
                    d.deleteTree(e.io, pkg) catch {};
                }
                var m = rt.Model.load(compiled, @enumFromInt(@intFromEnum(e.opts.compute_units))) catch |err| {
                    log.err("loading {s}: {s}", .{ compiled, rt.lastError() });
                    return err;
                };
                errdefer m.deinit();
                const arrays = try gpa.alloc(rt.MultiArray, g.inputs.len);
                var n: usize = 0;
                errdefer {
                    for (arrays[0..n]) |*a| a.deinit();
                    gpa.free(arrays);
                }
                const feeds = try gpa.alloc(rt.Model.Input, g.inputs.len);
                errdefer gpa.free(feeds);
                for (g.inputs) |in| {
                    arrays[n] = try rt.MultiArray.wrap(std.mem.sliceAsBytes(in.buf), in.shape, .float16);
                    feeds[n] = .{ .name = in.name, .array = arrays[n] };
                    n += 1;
                }
                g.cm = .{ .model = m, .arrays = arrays, .feeds = feeds };
            } else return error.CoreMLUnavailable,
        }
    }

    /// Cache key of a compiled program.
    fn graphKey(e: *const Engine, what: []const u8, ad: ?*const Adapter) u64 {
        var h = std.hash.Wyhash.init(0xc0de_4e1);
        h.update(what);
        h.update(std.mem.asBytes(&e.model_key));
        h.update(@embedFile("net.zig"));
        h.update(@tagName(e.opts.gelu));
        h.update(@tagName(e.opts.target));
        h.update(std.mem.asBytes(&@intFromEnum(e.opts.compute_units)));
        if (ad) |a| {
            h.update(std.mem.asBytes(&a.rank));
            h.update(std.mem.asBytes(&a.alpha));
            h.update(std.mem.sliceAsBytes(a.buf));
        }
        return h.final();
    }

    // ------------------------------------------------------------ inference

    /// Takes up to 30 s of 16 kHz audio: log-mel on the host, then the
    /// encoder program (audio encoder + every decoder layer's cross K/V).
    pub fn encode(e: *Engine, samples: []const f32) !void {
        const t0 = e.now();
        frontend.logMel(e.model, samples, e.mel32);
        for (e.mel16, e.mel32) |*d, v| d.* = @floatCast(v);
        const t1 = e.now();
        e.timing.mel_ms = ms(t0, t1);
        try e.enc.run(e.gpa);
        e.timing.encode_ms = e.since(t1);
        e.timing.steps_ms = 0;
        e.timing.steps = 0;
        e.have_logits = false;
    }

    /// Feeds one token at position `pos` through the decoder (KV-cached).
    /// The logits are always computed; `want_logits` is accepted for
    /// interface parity.
    pub fn step(e: *Engine, token: u32, pos: u32, want_logits: bool) !void {
        const hp = e.hp;
        const ctx = hp.n_text_ctx;
        const ts = hp.n_text_state;
        if (pos >= ctx or token >= hp.n_vocab) return error.OutOfRange;
        const m = e.model;
        for (e.x, m.tok_emb[token * ts ..][0..ts], m.dec_pos[pos * ts ..][0..ts]) |*d, t, p| d.* = @floatCast(@as(f32, @floatCast(t)) + p);
        for (e.mask, 0..) |*v, i| v.* = if (i < pos or i == ctx) 0 else @floatCast(net.mask_off);
        const t0 = e.now();
        try e.dec.run(e.gpa);
        e.timing.step_ms = e.since(t0);
        e.timing.steps_ms += e.timing.step_ms;
        e.timing.steps += 1;
        // This position's K/V into the cache ([heads][ctx][64]).
        const row = net.head_dim;
        for (0..hp.n_text_layer) |l| for (0..2) |w| {
            const src = e.newSlice(l, w);
            const dst = e.selfSlice(l, w);
            for (0..hp.n_text_head) |h| @memcpy(dst[(h * ctx + pos) * row ..][0..row], src[h * row ..][0..row]);
        };
        _ = want_logits;
        e.have_logits = true;
    }

    /// Picks the most likely token after the last `step`, excluding ids
    /// >= `ban_from` and `ban` (use `none`). Same semantics as the GPU's
    /// pick kernel: log-probs over the allowed tokens.
    pub fn pick(e: *Engine, ban_from: u32, ban: [2]u32, nosp: u32) !Pick {
        if (!e.have_logits) return error.NoLogits;
        const n: u32 = @intCast(e.hp.n_vocab);
        const allowed = struct {
            fn f(i: u32, bf: u32, b: [2]u32) bool {
                return i < bf and i != b[0] and i != b[1];
            }
        }.f;
        var best: u32 = 0;
        var bv: f32 = -std.math.inf(f32);
        var i: u32 = 0;
        while (i < n) : (i += 1) {
            if (!allowed(i, ban_from, ban)) continue;
            const v: f32 = @floatCast(e.logits[i]);
            if (v > bv) {
                bv = v;
                best = i;
            }
        }
        if (bv == -std.math.inf(f32)) return error.NoAllowedToken;
        var sum: f64 = 0;
        i = 0;
        while (i < n) : (i += 1) {
            if (allowed(i, ban_from, ban)) sum += @exp(@as(f64, @as(f32, @floatCast(e.logits[i])) - bv));
        }
        const lse: f32 = bv + @as(f32, @floatCast(@log(sum)));
        const nosp_ok = nosp < n and allowed(nosp, ban_from, ban);
        return .{
            .token = best,
            .logprob = bv - lse,
            .nosp_logprob = if (nosp_ok) @as(f32, @floatCast(e.logits[nosp])) - lse else -std.math.inf(f32),
        };
    }

    /// The last encode's log-mel spectrogram [n_mels][3000] (host output).
    pub fn melSpectrogram(e: *const Engine) []const f32 {
        return e.mel32;
    }

    /// Cross-attention keys/values of decoder layer `layer` from the last
    /// `encode`, as f32 [heads][1500][64] (k and v each).
    pub fn crossKV(e: *const Engine, layer: usize, k: []f32, v: []f32) void {
        for (k, e.crossSlice(layer, 0)) |*d, s| d.* = @floatCast(s);
        for (v, e.crossSlice(layer, 1)) |*d, s| d.* = @floatCast(s);
    }

    // ------------------------------------------------------------ adapter

    /// Rebuilds the decoder program with `ad` merged into its weights
    /// (W += scale·B·A, on the host), or the base weights with null. With
    /// Core ML that means a new package and a compile on the device (seconds,
    /// cached per adapter afterwards). `version` becomes adapter_version.
    pub fn setAdapter(e: *Engine, ad: ?*const Adapter, version: u32) !void {
        if (ad) |a| if (a.n_layers != e.hp.n_text_layer or a.n_state != e.hp.n_text_state) return error.AdapterModelMismatch;
        try e.openDecoder(ad, version);
    }

    // ------------------------------------------------------------ helpers

    fn now(e: *const Engine) Io.Timestamp {
        return Io.Clock.awake.now(e.io);
    }
    fn since(e: *const Engine, t0: Io.Timestamp) f64 {
        return ms(t0, e.now());
    }

    pub const none = gpu_engine.Engine.none;
};

fn ms(a: Io.Timestamp, b: Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(b.nanoseconds - a.nanoseconds)) / 1e6;
}

/// Program input/output names ("ck0", "sv3", ...), static strings.
const names = struct {
    const Kind = enum { ck, cv, sk, sv, kn, vn };
    const max_layers = 64;
    const table = blk: {
        @setEvalBranchQuota(100000);
        var t: [6][max_layers][]const u8 = undefined;
        for (@typeInfo(Kind).@"enum".field_names, 0..) |k, ki| for (0..max_layers) |l| {
            t[ki][l] = std.fmt.comptimePrint("{s}{d}", .{ k, l });
        };
        break :blk t;
    };
    fn get(k: Kind, l: usize) []const u8 {
        return table[@intFromEnum(k)][l];
    }
};

/// Interned small shapes (process lifetime).
const shapes = struct {
    var lock: std.atomic.Mutex = .unlocked;
    var list: [64][4]usize = undefined;
    var lens: [64]usize = undefined;
    var n: usize = 0;

    fn intern(s: []const usize) ![]const usize {
        while (!lock.tryLock()) std.atomic.spinLoopHint();
        defer lock.unlock();
        for (0..n) |i| if (std.mem.eql(usize, list[i][0..lens[i]], s)) return list[i][0..lens[i]];
        if (n == list.len or s.len > 4) return error.TooManyShapes;
        @memcpy(list[n][0..s.len], s);
        lens[n] = s.len;
        n += 1;
        return list[n - 1][0..s.len];
    }
};

/// Identifies the model's weights (for the compiled-program cache).
fn modelHash(m: *const Model) u64 {
    var h = std.hash.Wyhash.init(0x5717_c0de);
    h.update(std.mem.asBytes(&m.hp));
    h.update(std.mem.sliceAsBytes(m.mel_filters));
    h.update(std.mem.sliceAsBytes(m.conv1.w));
    h.update(std.mem.sliceAsBytes(m.conv2.w));
    h.update(std.mem.sliceAsBytes(m.enc_pos));
    for (m.enc_layers) |l| {
        h.update(std.mem.sliceAsBytes(l.attn.q.w));
        h.update(std.mem.sliceAsBytes(l.fc1.w));
        h.update(std.mem.sliceAsBytes(l.fc2.w));
    }
    h.update(std.mem.sliceAsBytes(m.tok_emb));
    for (m.dec_layers) |l| {
        h.update(std.mem.sliceAsBytes(l.attn.q.w));
        h.update(std.mem.sliceAsBytes(l.cross.k.w));
        h.update(std.mem.sliceAsBytes(l.fc1.w));
    }
    return h.final();
}
