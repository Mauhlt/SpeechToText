//! Runs the transformer on a Snapdragon X laptop's Hexagon NPU (HTP) through
//! Qualcomm's QNN API (bindings: ../TranslateNPU, module "qnn"), using the QNN
//! runtime already installed on the machine. Nothing from the SDK ships.
//!
//! Same interface as gpu/engine.zig's Engine, so the decoding loop in
//! transcriber.zig can drive either one:
//!
//!   init(gpa, model, opts) / deinit(gpa)
//!   encode(samples)                  up to 30 s of 16 kHz mono f32 audio
//!   step(token, pos, want_logits)    one decoder token (KV-cached)
//!   pick(ban_from, ban, nosp) Pick   greedy choice over the last logits
//!   setAdapter(?*const Adapter, version)
//!   deviceName()
//!
//! Two QNN graphs (see net.zig): the encoder (log-mel front end included)
//! and a one-token decoder step. Both are built op by op from QNN built-ins
//! and compiled by the backend at init; with `cache_dir` the compiled
//! contexts are saved there and reloaded next time (keyed by backend, model
//! and adapter). Everything runs on the calling thread: no worker threads.
//!
//! All model compute runs in the graphs: log-mel front end, encoder,
//! cross K/V, token + position embedding, decoder, output projection, and
//! the greedy pick (arg-max + log-softmax), so per token the host reads back
//! one token id and two log-probs. What stays on the host (single-threaded):
//!   * staging inputs: reflect-padding the audio into the input buffer and
//!     choosing its power-of-two FP16 scale (one pass over the frames), the
//!     attention mask row, the ban mask (only when the ban list changes);
//!   * copying each step's new K/V row into the cache buffer (QNN graphs are
//!     stateless; the cache goes back in as an input next step);
//!   * folding a LoRA into the decoder's f16 weights once per setAdapter
//!     (static weights must be host memory when the graph is built).
//! A step with want_logits is executed when `pick` supplies its ban list.
const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const qnn = @import("qnn");
const c = qnn.c;
const graph = @import("graph.zig");
const model_mod = @import("../model.zig");
const Model = model_mod.Model;
const adapter_mod = @import("../adapter.zig");
const Adapter = adapter_mod.Adapter;
const mel = @import("../mel.zig");
const gpu_engine = @import("../gpu/engine.zig");
/// The graph definitions (exposed for npu/dev).
pub const net = @import("net.zig");

const log = std.log.scoped(.npu);

pub const Pick = gpu_engine.Pick;

pub const Options = struct {
    /// Folder holding the QNN backend DLL (QnnHtp.dll / QairtHtp.dll), or
    /// null to search the machine the way npu-probe does.
    qnn_dir: ?[]const u8 = null,
    /// Writable folder for the compiled-graph cache (QNN context binary),
    /// e.g. the stt.Local profile dir. null = don't cache.
    cache_dir: ?[]const u8 = null,
    /// Use QNN's CPU backend instead of the HTP (development on x86).
    backend: enum { htp, cpu } = .htp,
};

/// Wall-clock time of the last calls (milliseconds).
pub const Timing = struct {
    /// Building + finalizing (or loading from cache) each graph at init / setAdapter.
    encoder_prepare_ms: f64 = 0,
    decoder_prepare_ms: f64 = 0,
    encoder_from_cache: bool = false,
    decoder_from_cache: bool = false,
    /// Last encode(): host preparation and graph execution.
    encode_host_ms: f64 = 0,
    encode_ms: f64 = 0,
    /// Last step() graph execution, and the sum over all steps since encode().
    step_ms: f64 = 0,
    steps_ms: f64 = 0,
    steps: u32 = 0,
};

const format_version = "stt-npu-1";

/// A compiled graph inside its own QNN context, plus the descriptions of
/// its input/output tensors (ids, names, shapes) used to execute it.
const Session = struct {
    context: c.Qnn_ContextHandle_t = null,
    graph: c.Qnn_GraphHandle_t = null,
    arena: ?*std.heap.ArenaAllocator = null,
    inputs: []c.Qnn_Tensor_t = &.{},
    outputs: []c.Qnn_Tensor_t = &.{},

    fn deinit(s: *Session, e: *Engine) void {
        if (s.context != null) if (e.api.contextFree) |f| {
            _ = f(s.context, null);
        };
        if (s.arena) |a| {
            a.deinit();
            e.gpa.destroy(a);
        }
        s.* = .{};
    }

    /// Takes deep copies of the I/O tensor descriptions.
    fn adopt(s: *Session, gpa: Allocator, ins: []const c.Qnn_Tensor_t, outs: []const c.Qnn_Tensor_t) !void {
        const arena = try gpa.create(std.heap.ArenaAllocator);
        arena.* = .init(gpa);
        s.arena = arena;
        const a = arena.allocator();
        s.inputs = try a.alloc(c.Qnn_Tensor_t, ins.len);
        s.outputs = try a.alloc(c.Qnn_Tensor_t, outs.len);
        for (s.inputs, ins) |*d, t| d.* = try copyTensor(a, t);
        for (s.outputs, outs) |*d, t| d.* = try copyTensor(a, t);
    }

    fn copyTensor(a: Allocator, t: c.Qnn_Tensor_t) !c.Qnn_Tensor_t {
        var r = t;
        const v = &r.unnamed_0.v1;
        v.name = (try a.dupeSentinel(u8, std.mem.span(t.unnamed_0.v1.name), 0)).ptr;
        v.dimensions = (try a.dupe(u32, t.unnamed_0.v1.dimensions[0..t.unnamed_0.v1.rank])).ptr;
        v.unnamed_0.clientBuf = .{ .data = null, .dataSize = 0 };
        return r;
    }
};

const What = union(enum) {
    encoder: net.EncoderStop,
    decoder: net.DecWeights,
};

/// Whether this build has the NPU engine: on for aarch64-windows (the
/// Snapdragon laptops) unless -Dnpu=false, off everywhere else. Off, QNN is
/// never compiled (no SDK needed) and `Engine.init` returns error.NpuUnavailable.
pub const enabled = @import("stt_options").npu;

pub const Engine = if (enabled) QnnEngine else NoNpu;

/// The engine of a build without the NPU: same calls as QnnEngine, but it
/// can never be created.
const NoNpu = struct {
    adapter_version: u32 = 0,

    pub fn init(gpa: Allocator, model: *const Model, opts: Options) !*NoNpu {
        _ = .{ gpa, model, opts };
        log.warn("this stt build has no NPU engine (it is built for aarch64-windows only)", .{});
        return error.NpuUnavailable;
    }
    pub fn deinit(e: *NoNpu, gpa: Allocator) void {
        _ = .{ e, gpa };
        unreachable;
    }
    pub fn deviceName(e: *const NoNpu) []const u8 {
        _ = e;
        unreachable;
    }
    pub fn encode(e: *NoNpu, samples_in: []const f32) !void {
        _ = .{ e, samples_in };
        unreachable;
    }
    pub fn step(e: *NoNpu, token: u32, pos: u32, want_logits: bool) !void {
        _ = .{ e, token, pos, want_logits };
        unreachable;
    }
    pub fn pick(e: *NoNpu, ban_from: u32, ban: [2]u32, nosp: u32) !Pick {
        _ = .{ e, ban_from, ban, nosp };
        unreachable;
    }
    pub fn setAdapter(e: *NoNpu, ad: ?*const Adapter, version: u32) !void {
        _ = .{ e, ad, version };
        unreachable;
    }
};

const QnnEngine = struct {
    /// Version tag of the adapter merged into the weights (0 = none).
    adapter_version: u32 = 0,

    gpa: Allocator,
    io: Io,
    model: *const Model,
    hp: model_mod.HParams,
    cfg: net.Config = .{},
    lib: qnn.Library,
    prov: qnn.Provider,
    api: *const qnn.Interface,
    logger: c.Qnn_LogHandle_t = null,
    backend: c.Qnn_BackendHandle_t = null,
    device: c.Qnn_DeviceHandle_t = null,
    htp: bool,
    dt: graph.DType,
    name: []u8,
    lib_path: []u8,
    cache_dir: ?[]u8,
    /// Hash of the backend identity (API versions, build, chip) for cache keys.
    backend_key: u64,
    model_key: u64 = 0,

    enc: Session = .{},
    dec: Session = .{},

    // Host buffers, bound to the graphs' tensors (in the network's type).
    audio: []u8,
    mel_bias: []u8,
    cross: []u8, // [layer][k|v][heads][1500][64]
    self_kv: []u8, // [layer][k|v][heads][ctx][64]
    mask: []u8,
    kv_new: []u8, // [layer][k|v][state]
    ban: []u8, // [n_vocab] additive mask for the pick
    token_in: []u8, // i32
    pos_in: []u8, // i32
    nosp_in: []u8, // i32
    pick_out: []u8, // i32
    lp: []u8,
    nlp: []u8,
    mel_out: []u8 = &.{}, // only for the .mel test graph
    padded: []f32,
    window: []f32,
    /// Decoder weights of the adapted layers with the current LoRA folded in
    /// (null = base weights, read straight from the model).
    merged: ?[]f16 = null,
    timing: Timing = .{},
    /// A step with want_logits waits here until `pick` supplies the ban
    /// list, so the graph can do the arg-max itself (see `pick`).
    pending: ?u32 = null, // its position
    /// The pick computed by the last executed step, and its ban list.
    last: ?struct { ban_from: u32, ban: [2]u32, nosp: u32, pos: u32, pick: Pick } = null,
    ban_key: ?struct { ban_from: u32, ban: [2]u32 } = null,

    pub fn init(gpa: Allocator, model: *const Model, opts: Options) !*Engine {
        return initWith(gpa, model, opts, .{});
    }

    /// Development knobs (see npu/dev).
    pub const DevOptions = struct {
        /// .mel: the encoder graph only computes the log-mel spectrogram
        /// (read with `melSpectrogram`), and there is no decoder.
        stop: net.EncoderStop = .full,
        cfg: net.Config = .{},
    };

    pub fn initWith(gpa: Allocator, model: *const Model, opts: Options, dev: DevOptions) !*Engine {
        const stop = dev.stop;
        if (builtin.os.tag != .windows) return error.NpuUnavailable;
        const hp = model.hp;
        if (hp.n_audio_state != hp.n_audio_head * net.head_dim or hp.n_text_state != hp.n_text_head * net.head_dim)
            return error.UnsupportedHeadDim;
        if (model.mel_filters.len != hp.n_mels * net.n_bins) return error.BadFilterbank;
        if (hp.n_audio_ctx * 2 != mel.n_frames) return error.UnsupportedAudioContext;

        const io = std.Io.Threaded.global_single_threaded.io();
        const htp = opts.backend == .htp;
        const names: []const []const u8 = if (htp) &.{ "QnnHtp.dll", "QairtHtp.dll" } else &.{ "QnnCpu.dll", "QairtCpu.dll" };
        const lib_path = (try qnn.findBackend(gpa, io, names, opts.qnn_dir)) orelse {
            log.warn("no {s} found (searched {s}, the exe folder, PATH, System32, the driver store, C:\\Qualcomm)", .{ names[0], opts.qnn_dir orelse "-" });
            return error.NpuUnavailable;
        };
        errdefer gpa.free(lib_path);
        // Deliberately never closed: backends may keep threads alive until exit.
        const lib = qnn.Library.open(lib_path) catch {
            log.warn("could not load {s}", .{lib_path});
            return error.NpuUnavailable;
        };
        const prov = lib.provider() catch return error.NpuUnavailable;

        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        const dt: graph.DType = if (htp) .f16 else .f32;
        const L = hp.n_text_layer;
        const A = hp.n_audio_ctx;
        const ts = hp.n_text_state;
        const esz = dt.size();
        e.* = .{
            .gpa = gpa,
            .io = io,
            .model = model,
            .hp = hp,
            .lib = lib,
            .prov = prov,
            .api = prov.api(),
            .htp = htp,
            .cfg = dev.cfg,
            .dt = dt,
            .name = &.{},
            .lib_path = lib_path,
            .cache_dir = null,
            .backend_key = 0,
            .audio = &.{},
            .mel_bias = &.{},
            .cross = &.{},
            .self_kv = &.{},
            .mask = &.{},
            .kv_new = &.{},
            .ban = &.{},
            .token_in = &.{},
            .pos_in = &.{},
            .nosp_in = &.{},
            .pick_out = &.{},
            .lp = &.{},
            .nlp = &.{},
            .padded = &.{},
            .window = &.{},
        };
        errdefer e.freeAll();
        if (opts.cache_dir) |d| e.cache_dir = try gpa.dupe(u8, d);

        e.audio = try gpa.alloc(u8, net.audio_rows * mel.hop * esz);
        e.mel_bias = try gpa.alloc(u8, esz);
        e.cross = try gpa.alloc(u8, L * 2 * A * ts * esz);
        e.self_kv = try gpa.alloc(u8, L * 2 * hp.n_text_ctx * ts * esz);
        @memset(e.self_kv, 0);
        e.mask = try gpa.alloc(u8, (hp.n_text_ctx + 1) * esz);
        e.kv_new = try gpa.alloc(u8, L * 2 * ts * esz);
        e.ban = try gpa.alloc(u8, hp.n_vocab * esz);
        @memset(e.ban, 0);
        inline for (.{ "token_in", "pos_in", "nosp_in", "pick_out" }) |f| {
            @field(e, f) = try gpa.alloc(u8, 4);
            @memset(@field(e, f), 0);
        }
        e.lp = try gpa.alloc(u8, esz);
        e.nlp = try gpa.alloc(u8, esz);
        e.padded = try gpa.alloc(f32, net.audio_rows * mel.hop);
        e.window = try gpa.alloc(f32, mel.n_fft);
        for (e.window, 0..) |*w, n| {
            w.* = @floatCast(0.5 * (1.0 - @cos(2.0 * std.math.pi * @as(f64, @floatFromInt(n)) / mel.n_fft)));
        }
        if (stop == .mel) e.mel_out = try gpa.alloc(u8, mel.n_frames * hp.n_mels * esz);

        try e.openBackend();

        const t0 = e.now();
        if (e.cache_dir != null) e.model_key = modelHash(model);
        e.enc = try e.openSession("stt_encoder", .{ .encoder = stop }, e.encKey(stop));
        e.timing.encoder_prepare_ms = e.since(t0);
        try e.bind(&e.enc);

        if (stop == .full) try e.openDecoder(null, 0);
        return e;
    }

    pub fn deinit(e: *Engine, gpa: Allocator) void {
        _ = gpa; // the engine's own allocator (the same one) owns everything
        e.freeAll();
        e.gpa.destroy(e);
    }

    fn freeAll(e: *Engine) void {
        const gpa = e.gpa;
        e.dec.deinit(e);
        e.enc.deinit(e);
        if (e.device != null) if (e.api.deviceFree) |f| {
            _ = f(e.device);
        };
        if (e.backend != null) if (e.api.backendFree) |f| {
            _ = f(e.backend);
        };
        if (e.logger != null) if (e.api.logFree) |f| {
            _ = f(e.logger);
        };
        e.device = null;
        e.backend = null;
        e.logger = null;
        inline for (.{ "audio", "mel_bias", "cross", "self_kv", "mask", "kv_new", "ban", "token_in", "pos_in", "nosp_in", "pick_out", "lp", "nlp", "mel_out", "name" }) |f| {
            gpa.free(@field(e, f));
            @field(e, f) = &.{};
        }
        gpa.free(e.padded);
        gpa.free(e.window);
        e.padded = &.{};
        e.window = &.{};
        if (e.merged) |m| gpa.free(m);
        e.merged = null;
        if (e.cache_dir) |d| gpa.free(d);
        e.cache_dir = null;
        gpa.free(e.lib_path);
        e.lib_path = &.{};
    }

    pub fn deviceName(e: *const Engine) []const u8 {
        return e.name;
    }

    /// Path of the QNN backend library in use.
    pub fn libraryPath(e: *const Engine) []const u8 {
        return e.lib_path;
    }

    // ------------------------------------------------------------ backend

    fn openBackend(e: *Engine) !void {
        const api = e.api;
        if (api.logCreate) |f| {
            if (f(logCallback, c.QNN_LOG_LEVEL_ERROR, &e.logger) & 0xffff != 0) e.logger = null;
        }
        try call(api.backendCreate.?(e.logger, null, &e.backend), "backendCreate");

        var h = std.hash.Wyhash.init(0);
        h.update(format_version);
        // Any change to the graph definitions invalidates cached graphs.
        h.update(@embedFile("net.zig"));
        h.update(@embedFile("graph.zig"));
        h.update(std.mem.asBytes(&e.prov.coreVersion()));
        h.update(std.mem.asBytes(&e.prov.backendVersion()));
        h.update(std.mem.asBytes(&e.prov.backendId()));
        if (e.prov.has("backendGetBuildId")) {
            var id: [*c]const u8 = null;
            if (api.backendGetBuildId.?(&id) == c.QNN_SUCCESS and id != null) h.update(std.mem.span(id));
        }

        var arch: u32 = 0;
        if (e.htp) {
            if (e.prov.has("deviceGetPlatformInfo")) {
                var info: [*c]const c.QnnDevice_PlatformInfo_t = null;
                if (api.deviceGetPlatformInfo.?(e.logger, &info) == c.QNN_SUCCESS and info != null) {
                    const v1 = info.*.unnamed_0.v1;
                    if (v1.numHwDevices > 0) {
                        const ext = v1.hwDevices[0].unnamed_0.v1.deviceInfoExtension;
                        if (ext != null and ext.*.devType == c.QNN_HTP_DEVICE_TYPE_ON_CHIP) {
                            const chip = ext.*.unnamed_0.onChipDevice;
                            arch = chip.arch;
                            h.update(std.mem.asBytes(&chip.arch));
                            h.update(std.mem.asBytes(&chip.socModel));
                        }
                    }
                    if (e.prov.has("deviceFreePlatformInfo")) _ = api.deviceFreePlatformInfo.?(e.logger, info);
                }
            }
            if (e.prov.has("deviceCreate")) try call(api.deviceCreate.?(e.logger, null, &e.device), "deviceCreate");
        }
        e.backend_key = h.final();
        e.name = if (e.htp)
            (if (arch != 0)
                try std.fmt.allocPrint(e.gpa, "Hexagon NPU (QNN HTP V{d})", .{arch})
            else
                try e.gpa.dupe(u8, "Hexagon NPU (QNN HTP)"))
        else
            try e.gpa.dupe(u8, "QNN CPU backend (reference)");
    }

    fn call(st: c.Qnn_ErrorHandle_t, what: []const u8) !void {
        qnn.check(st) catch |err| {
            log.err("QNN {s} failed (status {d})", .{ what, qnn.last_error });
            return err;
        };
    }

    // ------------------------------------------------------------ sessions

    fn encKey(e: *const Engine, stop: net.EncoderStop) u64 {
        var h = std.hash.Wyhash.init(1);
        h.update("enc");
        h.update(std.mem.asBytes(&e.backend_key));
        h.update(std.mem.asBytes(&e.model_key));
        h.update(std.mem.asBytes(&e.cfg));
        h.update(@tagName(stop));
        return h.final();
    }

    fn decKey(e: *const Engine, ad: ?*const Adapter, version: u32) u64 {
        var h = std.hash.Wyhash.init(2);
        h.update("dec");
        h.update(std.mem.asBytes(&e.backend_key));
        h.update(std.mem.asBytes(&e.model_key));
        h.update(std.mem.asBytes(&e.cfg));
        if (ad) |a| {
            h.update(std.mem.asBytes(&version));
            h.update(std.mem.asBytes(&a.rank));
            h.update(std.mem.asBytes(&a.alpha));
            h.update(std.mem.sliceAsBytes(a.buf));
        }
        return h.final();
    }

    fn cachePath(e: *const Engine, kind: []const u8, key: u64) !?[]u8 {
        const dir = e.cache_dir orelse return null;
        const file = try std.fmt.allocPrint(e.gpa, "npu-{s}-{x:0>16}-{x:0>16}.bin", .{ kind, e.model_key, key });
        defer e.gpa.free(file);
        return try std.fs.path.join(e.gpa, &.{ dir, file });
    }

    fn openSession(e: *Engine, name: [:0]const u8, what: What, key: u64) !Session {
        const kind = if (what == .encoder) "enc" else "dec";
        const path = try e.cachePath(kind, key);
        defer if (path) |p| e.gpa.free(p);
        if (path) |p| {
            if (e.loadSession(name, p, key)) |s_in| {
                var s = s_in;
                if (e.bind(&s)) {
                    e.setFromCache(what, true);
                    return s;
                } else |err| {
                    log.info("graph cache {s} does not match this build ({t}); rebuilding", .{ p, err });
                    s.deinit(e);
                }
            } else |err| switch (err) {
                error.FileNotFound => {},
                else => log.info("graph cache {s} not used: {t}", .{ p, err }),
            }
        }
        e.setFromCache(what, false);
        var s = try e.buildSession(name, what);
        errdefer s.deinit(e);
        if (path) |p| e.saveSession(&s, p, key, kind) catch |err| log.warn("could not write graph cache {s}: {t}", .{ p, err });
        return s;
    }

    fn setFromCache(e: *Engine, what: What, v: bool) void {
        switch (what) {
            .encoder => e.timing.encoder_from_cache = v,
            .decoder => e.timing.decoder_from_cache = v,
        }
    }

    fn graphConfigs(e: *const Engine, htp_cfg: *c.QnnHtpGraph_CustomConfig_t, graph_cfg: *c.QnnGraph_Config_t, list: *[2][*c]const c.QnnGraph_Config_t) [*c][*c]const c.QnnGraph_Config_t {
        if (!e.htp) return null;
        // Compute in FP16 (what the V73 HTP runs natively).
        htp_cfg.* = .{ .option = c.QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION };
        htp_cfg.unnamed_0.precision = c.QNN_PRECISION_FLOAT16;
        graph_cfg.* = .{ .option = c.QNN_GRAPH_CONFIG_OPTION_CUSTOM };
        graph_cfg.unnamed_0.customConfig = htp_cfg;
        list.* = .{ graph_cfg, null };
        return @ptrCast(list);
    }

    fn buildSession(e: *Engine, name: [:0]const u8, what: What) !Session {
        const api = e.api;
        var s: Session = .{};
        errdefer s.deinit(e);
        try call(api.contextCreate.?(e.backend, e.device, null, &s.context), "contextCreate");
        var htp_cfg: c.QnnHtpGraph_CustomConfig_t = undefined;
        var graph_cfg: c.QnnGraph_Config_t = undefined;
        var list: [2][*c]const c.QnnGraph_Config_t = undefined;
        try call(api.graphCreate.?(s.context, name.ptr, e.graphConfigs(&htp_cfg, &graph_cfg, &list), &s.graph), "graphCreate");

        var arena: std.heap.ArenaAllocator = .init(e.gpa);
        defer arena.deinit();
        var b: graph.Builder = .{ .api = api, .graph = s.graph, .arena = arena.allocator(), .dt = e.dt };
        switch (what) {
            .encoder => |stop| try net.encoder(&b, e.model, e.cfg, stop),
            .decoder => |dw| try net.decoder(&b, e.model, e.cfg, dw),
        }
        log.info("{s}: {d} nodes, {d} tensors; finalizing", .{ name, b.n_nodes, b.n_tensors });
        try call(api.graphFinalize.?(s.graph, null, null), "graphFinalize");
        try s.adopt(e.gpa, b.inputs.items, b.outputs.items);
        return s;
    }

    // Cache file: magic, key, tensor descriptions, context binary.
    const cache_magic = "STTNPUC1";

    fn saveSession(e: *Engine, s: *const Session, path: []const u8, key: u64, kind: []const u8) !void {
        if (!e.prov.has("contextGetBinarySize") or !e.prov.has("contextGetBinary")) return error.Unsupported;
        const api = e.api;
        var size: c.Qnn_ContextBinarySize_t = 0;
        try qnn.check(api.contextGetBinarySize.?(s.context, &size));
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(e.gpa);
        try out.appendSlice(e.gpa, cache_magic);
        try appendInt(&out, e.gpa, u64, key);
        for ([_][]const c.Qnn_Tensor_t{ s.inputs, s.outputs }) |list| {
            try appendInt(&out, e.gpa, u32, @intCast(list.len));
            for (list) |t| {
                const v = t.unnamed_0.v1;
                const nm = std.mem.span(v.name);
                try appendInt(&out, e.gpa, u32, v.id);
                try appendInt(&out, e.gpa, u32, @intCast(v.type));
                try appendInt(&out, e.gpa, u32, @intCast(v.dataType));
                try appendInt(&out, e.gpa, u32, v.rank);
                for (v.dimensions[0..v.rank]) |d| try appendInt(&out, e.gpa, u32, d);
                try appendInt(&out, e.gpa, u32, @intCast(nm.len));
                try out.appendSlice(e.gpa, nm);
            }
        }
        const head = out.items.len + 8;
        try out.resize(e.gpa, head + @as(usize, @intCast(size)));
        var written: c.Qnn_ContextBinarySize_t = 0;
        try qnn.check(api.contextGetBinary.?(s.context, out.items[head..].ptr, size, &written));
        std.mem.writeInt(u64, out.items[head - 8 ..][0..8], written, .little);
        out.shrinkRetainingCapacity(head + @as(usize, @intCast(written)));

        const cwd = Io.Dir.cwd();
        const dir = std.fs.path.dirname(path) orelse ".";
        cwd.createDirPath(e.io, dir) catch {};
        const tmp = try std.mem.concat(e.gpa, u8, &.{ path, ".tmp" });
        defer e.gpa.free(tmp);
        try cwd.writeFile(e.io, .{ .sub_path = tmp, .data = out.items });
        try cwd.rename(tmp, cwd, path, e.io);
        log.info("saved graph cache {s} ({d} MB)", .{ path, out.items.len >> 20 });
        e.pruneCache(kind, path);
    }

    /// Removes this model's other cache files of the same kind (an old
    /// adapter's decoder, an old backend's encoder).
    fn pruneCache(e: *Engine, kind: []const u8, keep: []const u8) void {
        const dir_path = e.cache_dir orelse return;
        var dir = Io.Dir.cwd().openDir(e.io, dir_path, .{ .iterate = true }) catch return;
        defer dir.close(e.io);
        var prefix_buf: [64]u8 = undefined;
        const prefix = std.fmt.bufPrint(&prefix_buf, "npu-{s}-{x:0>16}-", .{ kind, e.model_key }) catch return;
        const keep_base = std.fs.path.basename(keep);
        var victims: [16][128]u8 = undefined;
        var lens: [16]usize = undefined;
        var n: usize = 0;
        var it = dir.iterate();
        while (it.next(e.io) catch null) |ent| {
            if (ent.kind != .file or !std.mem.startsWith(u8, ent.name, prefix)) continue;
            if (std.mem.eql(u8, ent.name, keep_base) or ent.name.len > 128 or n == victims.len) continue;
            @memcpy(victims[n][0..ent.name.len], ent.name);
            lens[n] = ent.name.len;
            n += 1;
        }
        for (0..n) |i| dir.deleteFile(e.io, victims[i][0..lens[i]]) catch {};
    }

    fn loadSession(e: *Engine, name: [:0]const u8, path: []const u8, key: u64) !Session {
        const bytes = try Io.Dir.cwd().readFileAlloc(e.io, path, e.gpa, .limited(4 << 30));
        defer e.gpa.free(bytes);
        var r: Reader = .{ .buf = bytes };
        if (!std.mem.eql(u8, try r.take(8), cache_magic)) return error.BadCacheFile;
        if (try r.int(u64) != key) return error.CacheKeyMismatch;

        var s: Session = .{};
        errdefer s.deinit(e);
        const arena = try e.gpa.create(std.heap.ArenaAllocator);
        arena.* = .init(e.gpa);
        s.arena = arena;
        const a = arena.allocator();
        var lists: [2][]c.Qnn_Tensor_t = undefined;
        for (&lists) |*list| {
            const n = try r.int(u32);
            if (n > 4096) return error.BadCacheFile;
            list.* = try a.alloc(c.Qnn_Tensor_t, n);
            for (list.*) |*t| {
                const id = try r.int(u32);
                const ty = try r.int(u32);
                const dtype = try r.int(u32);
                const rank = try r.int(u32);
                if (rank > 8) return error.BadCacheFile;
                const dims = try a.alloc(u32, rank);
                for (dims) |*d| d.* = try r.int(u32);
                const nl = try r.int(u32);
                const nm = try a.dupeSentinel(u8, try r.take(nl), 0);
                const kind: qnn.TensorKind = switch (ty) {
                    c.QNN_TENSOR_TYPE_APP_WRITE => .app_write,
                    c.QNN_TENSOR_TYPE_APP_READ => .app_read,
                    else => return error.BadCacheFile,
                };
                t.* = qnn.tensor(nm.ptr, kind, @intCast(dtype), dims, null);
                t.unnamed_0.v1.id = id;
            }
        }
        s.inputs = lists[0];
        s.outputs = lists[1];
        const size = try r.int(u64);
        const bin = try r.take(@intCast(size));
        if (!e.prov.has("contextCreateFromBinary") or !e.prov.has("graphRetrieve")) return error.Unsupported;
        try qnn.check(e.api.contextCreateFromBinary.?(e.backend, e.device, null, bin.ptr, size, &s.context, null));
        try qnn.check(e.api.graphRetrieve.?(s.context, name.ptr, &s.graph));
        log.info("loaded graph {s} from cache {s}", .{ name, path });
        return s;
    }

    /// Points every input/output tensor of `s` at its host buffer.
    fn bind(e: *Engine, s: *Session) !void {
        for ([_][]c.Qnn_Tensor_t{ s.inputs, s.outputs }) |list| for (list) |*t| {
            const v = &t.unnamed_0.v1;
            const nm = std.mem.span(v.name);
            const buf = e.bufferFor(nm) orelse {
                log.warn("graph tensor {s} has no host buffer", .{nm});
                return error.BadGraph;
            };
            const is_int = v.dataType == c.QNN_DATATYPE_INT_32;
            var n: usize = if (is_int) 4 else e.dt.size();
            for (v.dimensions[0..v.rank]) |d| n *= d;
            if (n != buf.len or (!is_int and v.dataType != e.dt.qnnType())) {
                log.warn("graph tensor {s}: {d} bytes, host buffer {d}", .{ nm, n, buf.len });
                return error.BadGraph;
            }
            qnn.setBuffer(t, buf);
        };
    }

    fn bufferFor(e: *Engine, nm: []const u8) ?[]u8 {
        const esz = e.dt.size();
        const hp = e.hp;
        const per_cross = hp.n_audio_ctx * hp.n_text_state * esz;
        const per_self = hp.n_text_ctx * hp.n_text_state * esz;
        const per_new = hp.n_text_state * esz;
        if (std.mem.eql(u8, nm, "audio")) return e.audio;
        if (std.mem.eql(u8, nm, "mel_bias")) return e.mel_bias;
        if (std.mem.eql(u8, nm, "mel")) return e.mel_out;
        const named = .{
            .{ "mask", "mask" },   .{ "ban", "ban" },   .{ "token", "token_in" }, .{ "pos", "pos_in" },
            .{ "nosp", "nosp_in" }, .{ "pick", "pick_out" }, .{ "lp", "lp" },         .{ "nlp", "nlp" },
        };
        inline for (named) |p| if (std.mem.eql(u8, nm, p[0])) return @field(e, p[1]);
        if (nm.len < 3) return null;
        const layer = std.fmt.parseInt(usize, nm[2..], 10) catch return null;
        if (layer >= hp.n_text_layer) return null;
        // ck/cv/sk/sv: second letter is k|v; kn/vn: first letter is k|v.
        if (nm[1] == 'n') {
            const which: usize = if (nm[0] == 'k') 0 else if (nm[0] == 'v') 1 else return null;
            return e.kv_new[(layer * 2 + which) * per_new ..][0..per_new];
        }
        const which: usize = if (nm[1] == 'k') 0 else if (nm[1] == 'v') 1 else return null;
        const slot = layer * 2 + which;
        return switch (nm[0]) {
            'c' => e.cross[slot * per_cross ..][0..per_cross],
            's' => e.self_kv[slot * per_self ..][0..per_self],
            else => null,
        };
    }

    fn execute(e: *Engine, s: *const Session) !void {
        try call(e.api.graphExecute.?(s.graph, s.inputs.ptr, @intCast(s.inputs.len), s.outputs.ptr, @intCast(s.outputs.len), null, null), "graphExecute");
    }

    // ------------------------------------------------------------ inference

    /// Takes up to 30 s of 16 kHz audio, computes its log-mel spectrogram,
    /// runs the audio encoder and every decoder layer's cross-attention
    /// keys/values (all in the encoder graph).
    pub fn encode(e: *Engine, samples_in: []const f32) !void {
        const t0 = e.now();
        const samples = samples_in[0..@min(samples_in.len, mel.n_samples)];
        const n = samples.len;
        const total = mel.n_samples;
        const pad = mel.n_fft / 2;
        // Zero-pad to 30 s, then reflect-pad n_fft/2 on each side (as
        // kernels.pad_audio); only the samples the 3000 frames use.
        for (e.padded, 0..) |*p, idx| {
            const j: usize = if (idx < pad) pad - idx else if (idx >= pad + total) 2 * total + pad - 2 - idx else idx - pad;
            p.* = if (j < n) samples[j] else 0;
        }
        // FP16 range: scale by 2^k so the largest possible power value,
        // (Σ w|x|)² over a frame, is at most 2^15. Undone exactly in the
        // log domain through mel_bias.
        var amax: f32 = 0;
        for (0..mel.n_frames) |t| {
            var s: f32 = 0;
            for (e.padded[t * mel.hop ..][0..mel.n_fft], e.window) |x, w| s += w * @abs(x);
            amax = @max(amax, s);
        }
        const k: i32 = if (amax > 0) std.math.clamp(@as(i32, @intFromFloat(@floor(std.math.log2(181.0 / amax)))), -40, 40) else 0;
        const s = std.math.ldexp(@as(f32, 1), k);
        const shift = 2 * k + net.filterScaleLog2(e.model);
        const bias: f32 = -@as(f32, @floatFromInt(shift)) * 0.30102999566398120 / 4.0; // log10(2)
        e.store(e.mel_bias, 0, &.{bias});
        switch (e.dt) {
            .f16 => {
                const out = std.mem.bytesAsSlice(f16, e.audio);
                for (out, e.padded) |*o, v| o.* = @floatCast(v * s);
            },
            .f32 => {
                const out = std.mem.bytesAsSlice(f32, e.audio);
                for (out, e.padded) |*o, v| o.* = v * s;
            },
        }
        const t1 = e.now();
        e.timing.encode_host_ms = ms(t0, t1);
        try e.execute(&e.enc);
        e.timing.encode_ms = e.since(t1);
        e.timing.steps_ms = 0;
        e.timing.steps = 0;
        // A new window: an unpicked step of the old one is dropped.
        e.pending = null;
        e.last = null;
    }

    /// The last encode's log-mel spectrogram [3000][n_mels] (only for an
    /// engine made with initWith(.., .mel)).
    pub fn melSpectrogram(e: *const Engine, out: []f32) void {
        e.load(e.mel_out, out);
    }

    /// Cross-attention keys/values of decoder layer `layer` from the last
    /// `encode`, as f32 [heads][1500][64] (k and v each).
    pub fn crossKV(e: *const Engine, layer: usize, k: []f32, v: []f32) void {
        const per = e.hp.n_audio_ctx * e.hp.n_text_state * e.dt.size();
        e.load(e.cross[(layer * 2) * per ..][0..per], k);
        e.load(e.cross[(layer * 2 + 1) * per ..][0..per], v);
    }

    /// Replaces layer `layer`'s cross-attention K/V (f32 [heads][1500][64]),
    /// as if `encode` had produced them. For testing the decoder alone.
    pub fn setCrossKV(e: *Engine, layer: usize, k: []const f32, v: []const f32) void {
        const per = e.hp.n_audio_ctx * e.hp.n_text_state * e.dt.size();
        e.store(e.cross[(layer * 2) * per ..][0..per], 0, k);
        e.store(e.cross[(layer * 2 + 1) * per ..][0..per], 0, v);
        e.pending = null;
        e.last = null;
    }

    /// Feeds one token at position `pos` through the decoder (KV-cached).
    /// If `want_logits`, also keeps the vocabulary logits (read via `pick`).
    pub fn step(e: *Engine, token: u32, pos: u32, want_logits: bool) !void {
        const hp = e.hp;
        const ctx = hp.n_text_ctx;
        if (pos >= ctx or token >= hp.n_vocab) return error.OutOfRange;
        // A step still waiting for its pick must run first (its K/V row).
        if (e.pending) |p| try e.runStep(p);
        e.last = null;
        // The graph embeds the token and position itself (Gather).
        std.mem.writeInt(i32, e.token_in[0..4], @intCast(token), .little);
        std.mem.writeInt(i32, e.pos_in[0..4], @intCast(pos), .little);
        // Cached positions < pos are visible, the rest masked; the last
        // entry is this token (always visible).
        var mk: [4096]f32 = undefined;
        if (ctx + 1 > mk.len) return error.UnsupportedModelShape;
        for (mk[0 .. ctx + 1], 0..) |*v, i| v.* = if (i < pos or i == ctx) 0 else net.mask_off;
        e.store(e.mask, 0, mk[0 .. ctx + 1]);
        // With logits wanted, run when `pick` has supplied its ban list,
        // so the arg-max happens in the graph.
        if (want_logits) e.pending = pos else try e.runStep(pos);
    }

    /// Executes the decoder step whose inputs are staged, then writes this
    /// position's K/V into the cache ([heads][ctx][64]).
    fn runStep(e: *Engine, pos: u32) !void {
        const hp = e.hp;
        const ts = hp.n_text_state;
        const ctx = hp.n_text_ctx;
        e.pending = null;
        const t0 = e.now();
        try e.execute(&e.dec);
        e.timing.step_ms = e.since(t0);
        e.timing.steps_ms += e.timing.step_ms;
        e.timing.steps += 1;
        const esz = e.dt.size();
        const row = net.head_dim * esz;
        for (0..hp.n_text_layer * 2) |slot| {
            const src = e.kv_new[slot * ts * esz ..][0 .. ts * esz];
            const dst = e.self_kv[slot * ctx * ts * esz ..][0 .. ctx * ts * esz];
            for (0..hp.n_text_head) |h| {
                @memcpy(dst[(h * ctx + pos) * row ..][0..row], src[h * row ..][0..row]);
            }
        }
    }

    /// Picks the most likely token after the last `step(.., true)`,
    /// excluding ids >= `ban_from` and `ban` (use `none`). Same semantics as
    /// the GPU's pick kernel: log-probs over the allowed tokens. The arg-max
    /// and log-softmax run inside the decoder graph (that step executes now,
    /// with this ban list); the host reads back the token id and two
    /// log-probs.
    pub fn pick(e: *Engine, ban_from: u32, ban: [2]u32, nosp: u32) !Pick {
        const pos = if (e.pending) |p| p else if (e.last) |l| blk: {
            if (l.ban_from == ban_from and std.mem.eql(u32, &l.ban, &ban) and l.nosp == nosp) return l.pick;
            // Same step, other ban list: running it again is idempotent
            // (its own cache row is masked out of its inputs).
            break :blk l.pos;
        } else return error.NoLogits;

        const n: u32 = @intCast(e.hp.n_vocab);
        const changed = if (e.ban_key) |k| k.ban_from != ban_from or !std.mem.eql(u32, &k.ban, &ban) else true;
        if (changed) {
            // Additive mask: 0 = allowed, -30000 = banned (rewritten only
            // when the ban list changes: twice per window).
            switch (e.dt) {
                .f16 => for (std.mem.bytesAsSlice(f16, e.ban), 0..) |*v, i| {
                    v.* = if (i >= ban_from or i == ban[0] or i == ban[1]) net.mask_off else 0;
                },
                .f32 => for (std.mem.bytesAsSlice(f32, e.ban), 0..) |*v, i| {
                    v.* = if (i >= ban_from or i == ban[0] or i == ban[1]) net.mask_off else 0;
                },
            }
            e.ban_key = .{ .ban_from = ban_from, .ban = ban };
        }
        std.mem.writeInt(i32, e.nosp_in[0..4], @intCast(if (nosp < n) nosp else 0), .little);
        try e.runStep(pos);

        const tok: u32 = @bitCast(std.mem.readInt(i32, e.pick_out[0..4], .little));
        if (tok >= n) return error.NpuBadResult;
        var lp: [1]f32 = undefined;
        var nlp: [1]f32 = undefined;
        e.load(e.lp, &lp);
        e.load(e.nlp, &nlp);
        const banned = nosp >= ban_from or nosp == ban[0] or nosp == ban[1] or nosp >= n;
        const p: Pick = .{
            .token = tok,
            .logprob = lp[0],
            .nosp_logprob = if (banned) -std.math.inf(f32) else nlp[0],
        };
        e.last = .{ .ban_from = ban_from, .ban = ban, .nosp = nosp, .pos = pos, .pick = p };
        return p;
    }

    // ------------------------------------------------------------ adapter

    /// Restores the base decoder weights and, if `ad` is given, folds it in
    /// (W += scale·B·A, on the host, once), then rebuilds the decoder graph
    /// (or loads it from the cache). `version` is recorded as adapter_version.
    pub fn setAdapter(e: *Engine, ad: ?*const Adapter, version: u32) !void {
        if (ad) |a| if (a.n_layers != e.hp.n_text_layer or a.n_state != e.hp.n_text_state) return error.AdapterModelMismatch;
        try e.openDecoder(ad, version);
    }

    fn openDecoder(e: *Engine, ad: ?*const Adapter, version: u32) !void {
        const m = e.model;
        const L = e.hp.n_text_layer;
        const slots = try e.gpa.alloc([adapter_mod.n_slots][]const f16, L);
        defer e.gpa.free(slots);
        var merged: ?[]f16 = null;
        errdefer if (merged) |w| e.gpa.free(w);
        if (ad) |a| {
            var total: usize = 0;
            for (0..L) |l| for (0..adapter_mod.n_slots) |si| {
                total += adapter_mod.slotLinear(m, l, @enumFromInt(si)).w.len;
            };
            const buf = try e.gpa.alloc(f16, total);
            merged = buf;
            var off: usize = 0;
            const scale = a.scale();
            for (0..L) |l| for (0..adapter_mod.n_slots) |si| {
                const s: adapter_mod.Slot = @enumFromInt(si);
                const lin = adapter_mod.slotLinear(m, l, s);
                const lo = a.get(l, s);
                const w = buf[off..][0..lin.w.len];
                off += lin.w.len;
                // Same arithmetic as kernels.lora_merge.
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
        } else {
            for (0..L) |l| for (0..adapter_mod.n_slots) |si| {
                slots[l][si] = adapter_mod.slotLinear(m, l, @enumFromInt(si)).w;
            };
        }

        const t0 = e.now();
        var s = try e.openSession("stt_decoder", .{ .decoder = .{ .w = slots } }, e.decKey(ad, version));
        errdefer s.deinit(e);
        try e.bind(&s);
        e.timing.decoder_prepare_ms = e.since(t0);
        e.dec.deinit(e);
        e.dec = s;
        // Adapters change between windows; a half-done step is dropped.
        e.pending = null;
        e.last = null;
        // The weights now live in the compiled graph; the merged copy is
        // only needed while building.
        if (merged) |w| e.gpa.free(w);
        merged = null;
        e.adapter_version = if (ad != null) version else 0;
    }

    // ------------------------------------------------------------ helpers

    fn store(e: *const Engine, dst: []u8, at: usize, src: []const f32) void {
        switch (e.dt) {
            .f16 => {
                const d = std.mem.bytesAsSlice(f16, dst)[at..][0..src.len];
                for (d, src) |*o, v| o.* = @floatCast(v);
            },
            .f32 => {
                const d = std.mem.bytesAsSlice(f32, dst)[at..][0..src.len];
                @memcpy(d, src);
            },
        }
    }

    fn load(e: *const Engine, src: []const u8, dst: []f32) void {
        switch (e.dt) {
            .f16 => for (dst, std.mem.bytesAsSlice(f16, src)[0..dst.len]) |*o, v| {
                o.* = @floatCast(v);
            },
            .f32 => @memcpy(dst, std.mem.bytesAsSlice(f32, src)[0..dst.len]),
        }
    }

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

fn appendInt(out: *std.ArrayList(u8), gpa: Allocator, comptime I: type, v: I) !void {
    var b: [@sizeOf(I)]u8 = undefined;
    std.mem.writeInt(I, &b, v, .little);
    try out.appendSlice(gpa, &b);
}

const Reader = struct {
    buf: []const u8,
    pos: usize = 0,
    fn take(r: *Reader, n: usize) ![]const u8 {
        if (r.buf.len - r.pos < n) return error.BadCacheFile;
        defer r.pos += n;
        return r.buf[r.pos..][0..n];
    }
    fn int(r: *Reader, comptime I: type) !I {
        return std.mem.readInt(I, (try r.take(@sizeOf(I)))[0..@sizeOf(I)], .little);
    }
};

/// Identifies the model's weights (for the graph cache).
fn modelHash(m: *const Model) u64 {
    var h = std.hash.Wyhash.init(0x5717);
    h.update(std.mem.asBytes(&m.hp));
    h.update(std.mem.sliceAsBytes(m.mel_filters));
    const lin = struct {
        fn f(hh: *std.hash.Wyhash, l: model_mod.Linear) void {
            hh.update(std.mem.sliceAsBytes(l.w));
            if (l.b) |b| hh.update(std.mem.sliceAsBytes(b));
        }
        fn n(hh: *std.hash.Wyhash, x: model_mod.Norm) void {
            hh.update(std.mem.sliceAsBytes(x.w));
            hh.update(std.mem.sliceAsBytes(x.b));
        }
        fn a(hh: *std.hash.Wyhash, x: model_mod.Attn) void {
            f(hh, x.q);
            f(hh, x.k);
            f(hh, x.v);
            f(hh, x.o);
        }
    };
    lin.f(&h, m.conv1);
    lin.f(&h, m.conv2);
    h.update(std.mem.sliceAsBytes(m.enc_pos));
    for (m.enc_layers) |l| {
        lin.n(&h, l.attn_ln);
        lin.a(&h, l.attn);
        lin.n(&h, l.mlp_ln);
        lin.f(&h, l.fc1);
        lin.f(&h, l.fc2);
    }
    lin.n(&h, m.enc_ln_post);
    h.update(std.mem.sliceAsBytes(m.tok_emb));
    h.update(std.mem.sliceAsBytes(m.dec_pos));
    for (m.dec_layers) |l| {
        lin.n(&h, l.attn_ln);
        lin.a(&h, l.attn);
        lin.n(&h, l.cross_ln);
        lin.a(&h, l.cross);
        lin.n(&h, l.mlp_ln);
        lin.f(&h, l.fc1);
        lin.f(&h, l.fc2);
    }
    lin.n(&h, m.dec_ln);
    return h.final();
}

// QNN backend messages -> std.log.
const ntdll = struct {
    extern "ntdll" fn _vsnprintf(buf: [*]u8, n: usize, fmt: [*:0]const u8, args: c.va_list) callconv(.c) c_int;
};

fn logCallback(fmt: [*c]const u8, level: c.QnnLog_Level_t, timestamp: u64, args: c.va_list) callconv(.c) void {
    _ = timestamp;
    if (comptime builtin.os.tag != .windows) return;
    if (fmt == null) return;
    var buf: [1024]u8 = undefined;
    const r = ntdll._vsnprintf(&buf, buf.len - 1, fmt, args);
    const n: usize = if (r < 0) buf.len - 1 else @min(@as(usize, @intCast(r)), buf.len - 1);
    const msg = std.mem.trimEnd(u8, buf[0..n], "\r\n ");
    switch (level) {
        c.QNN_LOG_LEVEL_ERROR => log.err("qnn: {s}", .{msg}),
        c.QNN_LOG_LEVEL_WARN => log.warn("qnn: {s}", .{msg}),
        else => log.debug("qnn: {s}", .{msg}),
    }
}
