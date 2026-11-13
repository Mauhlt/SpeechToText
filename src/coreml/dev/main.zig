//! coreml-dev: checks the Core ML engine against the GPU engine, and
//! exports the generated programs.
//!
//!   (from SpeechToText) src/coreml/dev/zig-out/bin/coreml-dev [options]
//!
//!   --runtime coreml|reference  Core ML (Mac/iPhone; default there) or the
//!                       host reference interpreter (default elsewhere)
//!   --units ane|all|gpu|cpu     Core ML compute units (default ane =
//!                       cpu_and_neural_engine)
//!   --cache <dir>       compiled-program cache (Core ML; default
//!                       /tmp/stt-coreml)
//!   --model <path>      default models/ggml-tiny.en.bin
//!   --wav <path>        default testdata/hello.wav
//!   --gelu exact        erf GELU instead of the GPU's tanh form
//!   --no-fp16           reference runtime: plain f32 math
//!   --no-gpu            skip the GPU reference
//!   --adapter           also test setAdapter with a random LoRA
//!   --export <dir>      only write stt-encoder.mlpackage and
//!                       stt-decoder.mlpackage into <dir> (any OS) and exit
//!   --runs <n>          transcribe n times (timings of later runs are warm)
//!
//! Prints the log-mel and layer-0 cross K/V differences against the GPU,
//! both transcripts, and the timings.
const std = @import("std");
const Io = std.Io;
const stt = @import("stt");
const ce = stt.coreml.engine;
const lib = stt.coreml.lib;
const gpu = stt.gpu.engine;

const Args = struct {
    runtime: ce.Runtime = if (ce.has_coreml) .coreml else .reference,
    units: ce.ComputeUnits = .cpu_and_neural_engine,
    cache: []const u8 = "/tmp/stt-coreml",
    model: []const u8 = "models/ggml-tiny.en.bin",
    wav: []const u8 = "testdata/hello.wav",
    gelu: lib.mil.GeluMode = .tanh_approximation,
    fp16: bool = true,
    gpu: bool = true,
    adapter: bool = false,
    export_dir: ?[]const u8 = null,
    runs: usize = 1,
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
        if (std.mem.eql(u8, s, "--runtime")) {
            a.runtime = std.meta.stringToEnum(ce.Runtime, next) orelse return usage();
            i += 1;
        } else if (std.mem.eql(u8, s, "--units")) {
            a.units = if (std.mem.eql(u8, next, "ane")) .cpu_and_neural_engine else if (std.mem.eql(u8, next, "all")) .all else if (std.mem.eql(u8, next, "gpu")) .cpu_and_gpu else if (std.mem.eql(u8, next, "cpu")) .cpu_only else return usage();
            i += 1;
        } else if (std.mem.eql(u8, s, "--cache")) {
            a.cache = next;
            i += 1;
        } else if (std.mem.eql(u8, s, "--model")) {
            a.model = next;
            i += 1;
        } else if (std.mem.eql(u8, s, "--wav")) {
            a.wav = next;
            i += 1;
        } else if (std.mem.eql(u8, s, "--gelu")) {
            a.gelu = if (std.mem.eql(u8, next, "exact")) .exact else .tanh_approximation;
            i += 1;
        } else if (std.mem.eql(u8, s, "--export")) {
            a.export_dir = next;
            i += 1;
        } else if (std.mem.eql(u8, s, "--runs")) {
            a.runs = try std.fmt.parseInt(usize, next, 10);
            i += 1;
        } else if (std.mem.eql(u8, s, "--no-fp16")) {
            a.fp16 = false;
        } else if (std.mem.eql(u8, s, "--no-gpu")) {
            a.gpu = false;
        } else if (std.mem.eql(u8, s, "--adapter")) {
            a.adapter = true;
        } else return usage();
    }

    var model = try stt.Model.load(gpa, io, a.model);
    defer model.deinit();
    const hp = model.hp;
    std.debug.print("model {s}: state {d}, heads {d}, layers {d}/{d}\n", .{ a.model, hp.n_text_state, hp.n_text_head, hp.n_audio_layer, hp.n_text_layer });

    if (a.export_dir) |dir| return exportPackages(gpa, io, &model, dir, a.gelu);

    const samples = try stt.audio.load(gpa, io, a.wav, .{});
    defer gpa.free(samples);
    const A = hp.n_audio_ctx;
    const ts = hp.n_text_state;

    // ---- GPU reference
    var ref_k: []f32 = &.{};
    var ref_mel: []f32 = &.{};
    var ref_text: []u8 = &.{};
    defer gpa.free(ref_k);
    defer gpa.free(ref_mel);
    defer gpa.free(ref_text);
    if (a.gpu) gpu_ref: {
        var tr = stt.Transcriber.init(gpa, &model, .{}) catch |err| {
            std.debug.print("GPU reference unavailable: {t}\n", .{err});
            break :gpu_ref;
        };
        defer tr.deinit();
        const g = tr.engine.gpu;
        try g.encode(samples);
        try g.gpu.sync();
        ref_k = try gpa.alloc(f32, A * ts);
        try g.gpu.download(std.mem.sliceAsBytes(ref_k), g.crossKV(0).k);
        ref_mel = try gpa.alloc(f32, hp.n_mels * 3000);
        try g.gpu.download(std.mem.sliceAsBytes(ref_mel), g.b.spec);
        try g.gpu.sync();
        ref_text = try tr.transcribe(gpa, samples);
        std.debug.print("GPU {s}: \"{s}\"\n", .{ g.deviceName(), ref_text });
    }

    // ---- Core ML engine
    const t0 = Io.Clock.awake.now(io);
    var tr = try stt.Transcriber.init(gpa, &model, .{ .backend = .{ .coreml = .{
        .runtime = a.runtime,
        .cache_dir = a.cache,
        .compute_units = a.units,
        .gelu = a.gelu,
        .emulate_fp16 = a.fp16,
    } } });
    defer tr.deinit();
    const e = tr.engine.coreml;
    std.debug.print("{s} ready in {d:.0} ms: encoder {d:.0} ms{s}, decoder {d:.0} ms{s}\n", .{
        e.deviceName(),              msSince(io, t0),
        e.timing.encoder_prepare_ms, if (e.timing.encoder_from_cache) " (cache)" else "",
        e.timing.decoder_prepare_ms, if (e.timing.decoder_from_cache) " (cache)" else "",
    });

    try e.encode(samples);
    std.debug.print("encode: log-mel {d:.1} ms (host), encoder {d:.1} ms\n", .{ e.timing.mel_ms, e.timing.encode_ms });
    if (ref_mel.len > 0) {
        var max_err: f32 = 0;
        for (e.melSpectrogram(), ref_mel) |x, y| max_err = @max(max_err, @abs(x - y));
        std.debug.print("log-mel: max abs err vs GPU {d:.6}\n", .{max_err});
        const k = try gpa.alloc(f32, A * ts);
        defer gpa.free(k);
        const v = try gpa.alloc(f32, A * ts);
        defer gpa.free(v);
        e.crossKV(0, k, v);
        // Core ML [heads][1500][64], GPU [1500][heads*64]
        var me: f32 = 0;
        var mr: f32 = 0;
        var sum: f64 = 0;
        for (0..hp.n_text_head) |h| for (0..A) |t| for (0..64) |d| {
            const g = ref_k[t * ts + h * 64 + d];
            const n = k[(h * A + t) * 64 + d];
            me = @max(me, @abs(g - n));
            mr = @max(mr, @abs(g));
            sum += @abs(g - n);
        };
        std.debug.print("cross K layer 0: max abs err {d:.5} (max |ref| {d:.3}), mean {d:.6}\n", .{ me, mr, sum / @as(f64, @floatFromInt(k.len)) });
    }

    var text: []u8 = &.{};
    defer gpa.free(text);
    for (0..@max(a.runs, 1)) |r| {
        gpa.free(text);
        text = &.{};
        const t1 = Io.Clock.awake.now(io);
        text = try tr.transcribe(gpa, samples);
        std.debug.print("run {d}: {d:.1} ms wall (encoder {d:.1} ms, {d} steps {d:.1} ms, {d:.2} ms/step)\n", .{
            r + 1,         msSince(io, t1), e.timing.encode_ms, e.timing.steps, e.timing.steps_ms,
            e.timing.steps_ms / @as(f64, @floatFromInt(@max(e.timing.steps, 1))),
        });
    }
    std.debug.print("Core ML: \"{s}\"\n", .{text});
    if (ref_text.len > 0) std.debug.print("transcript {s} the GPU's\n", .{if (std.mem.eql(u8, text, ref_text)) "MATCHES" else "DIFFERS from"});

    if (a.adapter) {
        var ad = try stt.Adapter.init(gpa, &model, 8, 16, 1);
        defer ad.deinit(gpa);
        for (ad.loras) |lo| for (lo.b, 0..) |*x, j| {
            x.* = 0.002 * @sin(@as(f32, @floatFromInt(j)) * 0.37);
        };
        const t2 = Io.Clock.awake.now(io);
        try tr.setAdapter(&ad, 7);
        std.debug.print("setAdapter: {d:.0} ms (decoder {d:.0} ms{s}), version {d}\n", .{ msSince(io, t2), e.timing.decoder_prepare_ms, if (e.timing.decoder_from_cache) " (cache)" else "", e.adapter_version });
        const t3 = try tr.transcribe(gpa, samples);
        defer gpa.free(t3);
        std.debug.print("  with adapter: \"{s}\"\n", .{t3});
        try tr.setAdapter(null, 0);
        const t4 = try tr.transcribe(gpa, samples);
        defer gpa.free(t4);
        std.debug.print("  base again:   \"{s}\" ({s})\n", .{ t4, if (std.mem.eql(u8, t4, text)) "same as before" else "DIFFERENT" });
    }
}

fn exportPackages(gpa: std.mem.Allocator, io: Io, m: *const stt.Model, dir: []const u8, gelu: lib.mil.GeluMode) !void {
    const cwd = Io.Dir.cwd();
    try cwd.createDirPath(io, dir);
    var d = try cwd.openDir(io, dir, .{});
    defer d.close(io);
    {
        var b = try lib.mil.Builder.init(gpa, .{ .target = .ios17, .description = "stt encoder" });
        defer b.deinit();
        try stt.coreml.net.encoder(&b, m, .{ .gelu = gelu });
        try lib.package.write(gpa, io, d, "stt-encoder.mlpackage", &b);
        std.debug.print("{s}/stt-encoder.mlpackage: {d} ops\n", .{ dir, b.ops.items.len });
    }
    {
        const slots = try stt.coreml.net.baseWeights(gpa, m);
        defer gpa.free(slots);
        var b = try lib.mil.Builder.init(gpa, .{ .target = .ios17, .description = "stt decoder step" });
        defer b.deinit();
        try stt.coreml.net.decoder(&b, m, .{ .gelu = gelu }, .{ .w = slots });
        try lib.package.write(gpa, io, d, "stt-decoder.mlpackage", &b);
        std.debug.print("{s}/stt-decoder.mlpackage: {d} ops\n", .{ dir, b.ops.items.len });
    }
}

fn usage() error{Usage} {
    std.debug.print("usage: coreml-dev [--runtime coreml|reference] [--units ane|all|gpu|cpu] [--cache dir] [--model path] [--wav path] [--gelu exact] [--no-fp16] [--no-gpu] [--adapter] [--export dir] [--runs n]\n", .{});
    return error.Usage;
}

fn msSince(io: Io, t0: Io.Timestamp) f64 {
    const t1 = Io.Clock.awake.now(io);
    return @as(f64, @floatFromInt(t1.nanoseconds - t0.nanoseconds)) / 1e6;
}
