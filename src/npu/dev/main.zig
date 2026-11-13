//! npu-dev: checks the NPU engine against the GPU engine.
//!
//!   cd SpeechToText\src\npu\dev && zig build
//!   (from SpeechToText) src\npu\dev\zig-out\bin\npu-dev.exe [options]
//!
//!   --backend cpu|htp   QNN backend (default cpu)
//!   --qnn <dir>         folder with the backend DLL (default: search)
//!   --model <path>      default models/ggml-tiny.en.bin
//!   --wav <path>        default testdata/hello.wav
//!   --cache <dir>       graph cache folder (default: none)
//!   --mel               also compare the log-mel spectrogram (extra graph)
//!   --no-gpu            skip the GPU reference
//!   --gelu builtin      use QNN's Gelu op instead of the tanh form
//!   --adapter           also test setAdapter with a random LoRA
//!   --prepare-only      only build/compile the graphs (no execution)
//!   --hybrid <n>        encoder on the CPU backend, then <n> decoder tokens on
//!                       --backend (for the slow x86 HTP simulator)
//!
//! Compares layer 0's cross-attention K/V (and optionally the mel
//! spectrogram) and runs a full greedy decode on both engines.
const std = @import("std");
const Io = std.Io;
const stt = @import("stt");
const npu = stt.npu.engine;
const net = npu.net;
const gpu = stt.gpu.engine;

const Args = struct {
    backend: @FieldType(npu.Options, "backend") = .cpu,
    qnn: ?[]const u8 = null,
    model: []const u8 = "models/ggml-tiny.en.bin",
    wav: []const u8 = "testdata/hello.wav",
    cache: ?[]const u8 = null,
    mel: bool = false,
    gpu: bool = true,
    gelu: net.Gelu = .tanh,
    adapter: bool = false,
    prepare_only: bool = false,
    hybrid: usize = 0,
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    var a: Args = .{};
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const s = argv[i];
        const next = if (i + 1 < argv.len) argv[i + 1] else "";
        if (std.mem.eql(u8, s, "--backend")) {
            a.backend = if (std.mem.eql(u8, next, "htp")) .htp else .cpu;
            i += 1;
        } else if (std.mem.eql(u8, s, "--qnn")) {
            a.qnn = next;
            i += 1;
        } else if (std.mem.eql(u8, s, "--model")) {
            a.model = next;
            i += 1;
        } else if (std.mem.eql(u8, s, "--wav")) {
            a.wav = next;
            i += 1;
        } else if (std.mem.eql(u8, s, "--cache")) {
            a.cache = next;
            i += 1;
        } else if (std.mem.eql(u8, s, "--gelu")) {
            a.gelu = if (std.mem.eql(u8, next, "builtin")) .builtin else .tanh;
            i += 1;
        } else if (std.mem.eql(u8, s, "--mel")) {
            a.mel = true;
        } else if (std.mem.eql(u8, s, "--no-gpu")) {
            a.gpu = false;
        } else if (std.mem.eql(u8, s, "--hybrid")) {
            a.hybrid = try std.fmt.parseInt(usize, next, 10);
            i += 1;
        } else if (std.mem.eql(u8, s, "--prepare-only")) {
            a.prepare_only = true;
        } else if (std.mem.eql(u8, s, "--adapter")) {
            a.adapter = true;
        } else {
            std.debug.print("unknown argument {s}\n", .{s});
            std.process.exit(2);
        }
    }

    var model = try stt.Model.load(gpa, io, a.model);
    defer model.deinit();
    const samples = try stt.audio.load(gpa, io, a.wav, .{});
    defer gpa.free(samples);
    const hp = model.hp;
    const A = hp.n_audio_ctx;
    const ts = hp.n_text_state;
    std.debug.print("model {s}: state {d}, heads {d}, layers {d}/{d}; audio {d:.2} s\n", .{ a.model, ts, hp.n_text_head, hp.n_audio_layer, hp.n_text_layer, @as(f64, @floatFromInt(samples.len)) / 16000.0 });

    // ---- GPU reference
    var ref_k: []f32 = &.{};
    var ref_v: []f32 = &.{};
    var ref_mel: []f32 = &.{};
    var ref_text: []u8 = &.{};
    defer gpa.free(ref_k);
    defer gpa.free(ref_v);
    defer gpa.free(ref_mel);
    defer gpa.free(ref_text);
    if (a.gpu) gpu_ref: {
        const g = gpu.Engine.init(gpa, &model) catch |err| {
            std.debug.print("GPU reference unavailable: {t}\n", .{err});
            break :gpu_ref;
        };
        defer g.deinit(gpa);
        try g.encode(samples);
        try g.gpu.sync();
        ref_k = try gpa.alloc(f32, A * ts);
        ref_v = try gpa.alloc(f32, A * ts);
        const kv = g.crossKV(0);
        try g.gpu.download(std.mem.sliceAsBytes(ref_k), kv.k);
        try g.gpu.download(std.mem.sliceAsBytes(ref_v), kv.v);
        ref_mel = try gpa.alloc(f32, hp.n_mels * 3000);
        try g.gpu.download(std.mem.sliceAsBytes(ref_mel), g.b.spec);
        try g.gpu.sync();
        const t0 = Io.Clock.awake.now(io);
        ref_text = try decode(gpa, g, &model);
        const t = g.timings();
        std.debug.print("GPU {s}: encode {d:.1} ms + cross {d:.1} ms; decode {d:.1} ms wall\n  \"{s}\"\n", .{ g.deviceName(), t.encode_ms, t.cross_ms, msSince(io, t0), ref_text });
    }

    const opts: npu.Options = .{ .backend = a.backend, .qnn_dir = a.qnn, .cache_dir = a.cache };
    if (a.hybrid > 0) return hybrid(gpa, io, &model, samples, opts, a.hybrid);

    // ---- optional: log-mel only graph
    if (a.mel) {
        const t0 = Io.Clock.awake.now(io);
        const e = try npu.Engine.initWith(gpa, &model, opts, .{ .stop = .mel, .cfg = .{ .gelu = a.gelu } });
        defer e.deinit(gpa);
        std.debug.print("NPU mel graph ready in {d:.0} ms\n", .{msSince(io, t0)});
        try e.encode(samples);
        const out = try gpa.alloc(f32, 3000 * hp.n_mels);
        defer gpa.free(out);
        e.melSpectrogram(out);
        if (ref_mel.len > 0) {
            // GPU layout [mel][frame], NPU [frame][mel]
            var max_err: f32 = 0;
            var sum_err: f64 = 0;
            var worst: usize = 0;
            for (0..3000) |t| for (0..hp.n_mels) |m| {
                const d = @abs(out[t * hp.n_mels + m] - ref_mel[m * 3000 + t]);
                sum_err += d;
                if (d > max_err) {
                    max_err = d;
                    worst = t * hp.n_mels + m;
                }
            };
            std.debug.print("mel: max abs err {d:.5} (frame {d} mel {d}: npu {d:.4} gpu {d:.4}), mean {d:.6}\n", .{ max_err, worst / hp.n_mels, worst % hp.n_mels, out[worst], ref_mel[(worst % hp.n_mels) * 3000 + worst / hp.n_mels], sum_err / @as(f64, @floatFromInt(out.len)) });
        } else std.debug.print("mel[0..4] = {any}\n", .{out[0..4]});
    }

    // ---- NPU engine
    var t0 = Io.Clock.awake.now(io);
    const e = try npu.Engine.initWith(gpa, &model, opts, .{ .cfg = .{ .gelu = a.gelu } });
    defer e.deinit(gpa);
    std.debug.print("NPU {s} ({s}) ready in {d:.0} ms: encoder {d:.0} ms{s}, decoder {d:.0} ms{s}\n", .{
        e.deviceName(),                 e.libraryPath(),                                               msSince(io, t0),
        e.timing.encoder_prepare_ms,    if (e.timing.encoder_from_cache) " (cache)" else "",          e.timing.decoder_prepare_ms,
        if (e.timing.decoder_from_cache) " (cache)" else "",
    });
    if (a.prepare_only) return;

    try e.encode(samples);
    std.debug.print("NPU encode: host {d:.1} ms, graph {d:.1} ms\n", .{ e.timing.encode_host_ms, e.timing.encode_ms });
    {
        const k = try gpa.alloc(f32, A * ts);
        defer gpa.free(k);
        const v = try gpa.alloc(f32, A * ts);
        defer gpa.free(v);
        e.crossKV(0, k, v);
        if (ref_k.len > 0) {
            for ([_][]const f32{ k, v }, [_][]const f32{ ref_k, ref_v }, [_][]const u8{ "K", "V" }) |got, want, nm| {
                // NPU [heads][1500][64], GPU [1500][heads*64]
                var max_err: f32 = 0;
                var max_ref: f32 = 0;
                var sum_err: f64 = 0;
                for (0..hp.n_text_head) |h| for (0..A) |t| for (0..64) |d| {
                    const g = want[t * ts + h * 64 + d];
                    const n = got[(h * A + t) * 64 + d];
                    max_err = @max(max_err, @abs(g - n));
                    max_ref = @max(max_ref, @abs(g));
                    sum_err += @abs(g - n);
                };
                std.debug.print("cross {s} layer 0: max abs err {d:.5} (max |ref| {d:.3}), mean abs err {d:.6}\n", .{ nm, max_err, max_ref, sum_err / @as(f64, @floatFromInt(got.len)) });
            }
        } else std.debug.print("cross K[0..4] = {any}\n", .{k[0..4]});
    }

    t0 = Io.Clock.awake.now(io);
    const text = try decode(gpa, e, &model);
    defer gpa.free(text);
    std.debug.print("NPU decode: {d} steps, {d:.1} ms total ({d:.2} ms/step graph), {d:.1} ms wall\n  \"{s}\"\n", .{ e.timing.steps, e.timing.steps_ms, e.timing.steps_ms / @as(f64, @floatFromInt(@max(e.timing.steps, 1))), msSince(io, t0), text });
    if (ref_text.len > 0) std.debug.print("transcript {s} the GPU's\n", .{if (std.mem.eql(u8, text, ref_text)) "MATCHES" else "DIFFERS from"});

    if (a.adapter) {
        var ad = try stt.Adapter.init(gpa, &model, 8, 16, 1);
        defer ad.deinit(gpa);
        // Small nonzero B so the merge changes the weights a little.
        for (ad.loras) |lo| for (lo.b, 0..) |*x, j| {
            x.* = 0.002 * @sin(@as(f32, @floatFromInt(j)) * 0.37);
        };
        t0 = Io.Clock.awake.now(io);
        try e.setAdapter(&ad, 7);
        std.debug.print("setAdapter: {d:.0} ms (decoder {d:.0} ms{s}), version {d}\n", .{ msSince(io, t0), e.timing.decoder_prepare_ms, if (e.timing.decoder_from_cache) " (cache)" else "", e.adapter_version });
        const t2 = try decode(gpa, e, &model);
        defer gpa.free(t2);
        std.debug.print("  with adapter: \"{s}\"\n", .{t2});
        try e.setAdapter(null, 0);
        const t3 = try decode(gpa, e, &model);
        defer gpa.free(t3);
        std.debug.print("  base again:   \"{s}\" ({s})\n", .{ t3, if (std.mem.eql(u8, t3, text)) "same as before" else "DIFFERENT" });
    }
}

/// Encoder on QNN's CPU backend, then the first `n` greedy tokens on both
/// the CPU engine and `opts.backend` (fed the same cross K/V).
fn hybrid(gpa: std.mem.Allocator, io: Io, m: *const stt.Model, samples: []const f32, opts: npu.Options, n: usize) !void {
    const hp = m.hp;
    var cpu_opts = opts;
    cpu_opts.backend = .cpu;
    cpu_opts.cache_dir = null;
    const ref = try npu.Engine.init(gpa, m, cpu_opts);
    defer ref.deinit(gpa);
    try ref.encode(samples);
    const want = try decodeTokens(gpa, ref, m, n);
    defer gpa.free(want.toks);
    defer gpa.free(want.lps);

    var t0 = Io.Clock.awake.now(io);
    const e = try npu.Engine.init(gpa, m, opts);
    defer e.deinit(gpa);
    std.debug.print("{s} ready in {d:.0} ms (encoder {d:.0} ms{s}, decoder {d:.0} ms{s})\n", .{ e.deviceName(), msSince(io, t0), e.timing.encoder_prepare_ms, if (e.timing.encoder_from_cache) " cache" else "", e.timing.decoder_prepare_ms, if (e.timing.decoder_from_cache) " cache" else "" });
    const per = hp.n_audio_ctx * hp.n_text_state;
    const k = try gpa.alloc(f32, per);
    defer gpa.free(k);
    const v = try gpa.alloc(f32, per);
    defer gpa.free(v);
    for (0..hp.n_text_layer) |l| {
        ref.crossKV(l, k, v);
        e.setCrossKV(l, k, v);
    }
    t0 = Io.Clock.awake.now(io);
    const got = try decodeTokens(gpa, e, m, n);
    defer gpa.free(got.toks);
    defer gpa.free(got.lps);
    std.debug.print("{d} steps in {d:.0} ms ({d:.1} ms/step graph)\n", .{ e.timing.steps, msSince(io, t0), e.timing.steps_ms / @as(f64, @floatFromInt(@max(e.timing.steps, 1))) });
    var max_lp: f32 = 0;
    for (want.toks, 0..) |t, i| {
        const g = if (i < got.toks.len) got.toks[i] else 0;
        const gl = if (i < got.lps.len) got.lps[i] else 0;
        max_lp = @max(max_lp, @abs(gl - want.lps[i]));
        std.debug.print("  {d}: cpu {d} ({s}) lp {d:.4} | {s} {d} lp {d:.4}{s}\n", .{ i, t, m.vocab[@min(t, m.vocab.len - 1)], want.lps[i], @tagName(opts.backend), g, gl, if (g != t) "  <-- differs" else "" });
    }
    std.debug.print("tokens {s}; max |logprob diff| {d:.4}\n", .{ if (std.mem.eql(u32, want.toks, got.toks)) "MATCH" else "DIFFER", max_lp });
}

const Toks = struct { toks: []u32, lps: []f32 };

/// The first `n` greedy picks (index 0 = no-speech log-prob after SOT).
fn decodeTokens(gpa: std.mem.Allocator, e: anytype, m: *const stt.Model, n: usize) !Toks {
    const tk = m.tokens;
    const none = gpu.Engine.none;
    var toks: std.ArrayList(u32) = .empty;
    errdefer toks.deinit(gpa);
    var lps: std.ArrayList(f32) = .empty;
    errdefer lps.deinit(gpa);
    const n_vocab: u32 = @intCast(m.hp.n_vocab);
    try e.step(tk.sot, 0, true);
    const ns = try e.pick(n_vocab, .{ none, none }, tk.nosp);
    try toks.append(gpa, ns.token);
    try lps.append(gpa, ns.nosp_logprob);
    try e.step(tk.not, 1, true);
    var pos: u32 = 2;
    while (toks.items.len < n) {
        const ban: [2]u32 = if (toks.items.len == 1) .{ tk.eot, 220 } else .{ none, none };
        const p = try e.pick(tk.eot + 1, ban, tk.nosp);
        try toks.append(gpa, p.token);
        try lps.append(gpa, p.logprob);
        if (p.token == tk.eot) break;
        try e.step(p.token, pos, true);
        pos += 1;
    }
    return .{ .toks = try toks.toOwnedSlice(gpa), .lps = try lps.toOwnedSlice(gpa) };
}

fn msSince(io: Io, t0: Io.Timestamp) f64 {
    const t1 = Io.Clock.awake.now(io);
    return @as(f64, @floatFromInt(t1.nanoseconds - t0.nanoseconds)) / 1e6;
}

/// Greedy decoding of the last encoded window, as transcriber.zig does it
/// (no prompt, no loop detection).
fn decode(gpa: std.mem.Allocator, e: anytype, m: *const stt.Model) ![]u8 {
    const tk = m.tokens;
    const none = gpu.Engine.none;
    var prompt: std.ArrayList(u32) = .empty;
    defer prompt.deinit(gpa);
    try prompt.append(gpa, tk.sot);
    if (tk.multilingual) {
        try prompt.append(gpa, tk.lang_en);
        try prompt.append(gpa, tk.transcribe);
    }
    try prompt.append(gpa, tk.not);
    const n_vocab: u32 = @intCast(m.hp.n_vocab);
    var pos: u32 = 0;
    for (prompt.items, 0..) |tok, i| {
        try e.step(tok, pos, i == 0 or i == prompt.items.len - 1);
        pos += 1;
        if (i == 0) _ = try e.pick(n_vocab, .{ none, none }, tk.nosp);
    }
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(gpa);
    while (out.items.len < 224) {
        const ban: [2]u32 = if (out.items.len == 0) .{ tk.eot, 220 } else .{ none, none };
        const p = try e.pick(tk.eot + 1, ban, tk.nosp);
        if (p.token == tk.eot) break;
        try out.append(gpa, p.token);
        try e.step(p.token, pos, true);
        pos += 1;
    }
    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(gpa);
    for (out.items) |t| if (t < m.vocab.len) try text.appendSlice(gpa, m.vocab[t]);
    return text.toOwnedSlice(gpa);
}
