//! GPU integration tests. They need models/ggml-tiny.en.bin,
//! testdata/hello.wav and an NVIDIA GPU, and are skipped otherwise.
const std = @import("std");
const Io = std.Io;
const Model = @import("model.zig").Model;
const Transcriber = @import("transcriber.zig").Transcriber;
const Adapter = @import("adapter.zig").Adapter;
const trainer = @import("gpu/trainer.zig");
const audio = @import("audio.zig");
const home_mod = @import("home.zig");
const schedule = @import("schedule.zig");

const model_path = "models/ggml-tiny.en.bin";
const text = "Hello world. This is a test of speech recognition written in pure Zig.";

fn loadModel() !Model {
    return Model.load(std.testing.allocator, std.testing.io, model_path) catch error.SkipZigTest;
}

fn loadHello() ![]f32 {
    return audio.load(std.testing.allocator, std.testing.io, "testdata/hello.wav", .{}) catch error.SkipZigTest;
}

test "gpu: transcribe" {
    const gpa = std.testing.allocator;
    var model = try loadModel();
    defer model.deinit();
    const samples = try loadHello();
    defer gpa.free(samples);
    var tr = Transcriber.init(gpa, &model, .{}) catch return error.SkipZigTest;
    defer tr.deinit();
    const out = try tr.transcribe(gpa, samples);
    defer gpa.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "Hello") != null);
}

test "gpu: LoRA gradients match finite differences" {
    const gpa = std.testing.allocator;
    var model = try loadModel();
    defer model.deinit();
    const samples = try loadHello();
    defer gpa.free(samples);
    var ad = try Adapter.init(gpa, &model, 8, 16, 1);
    defer ad.deinit(gpa);
    for (ad.loras) |lo| for (lo.b, 0..) |*x, i| {
        x.* = 0.02 * @sin(@as(f32, @floatFromInt(i)) * 0.37);
    };
    const s: trainer.Sample = .{ .audio = samples, .text = text };
    const grad = try gpa.alloc(f32, ad.buf.len);
    defer gpa.free(grad);
    _ = trainer.lossAndGrad(gpa, &model, &ad, s, grad) catch |err| switch (err) {
        error.CudaUnavailable, error.CudaNoDevice => return error.SkipZigTest,
        else => return err,
    };
    // Check the largest-gradient parameters of a few slots numerically.
    const slots = [_]@import("adapter.zig").Slot{ .q, .cross_o, .fc2 };
    for (slots) |sl| {
        const lo = ad.get(1, sl);
        const off = ad.offsetOf(1, sl) + lo.a.len; // first entries of B
        var best: usize = off;
        for (off..off + lo.b.len) |i| {
            if (@abs(grad[i]) > @abs(grad[best])) best = i;
        }
        const eps = 2e-2;
        const orig = ad.buf[best];
        ad.buf[best] = orig + eps;
        const lp = try trainer.evalLoss(gpa, &model, &ad, s);
        ad.buf[best] = orig - eps;
        const lm = try trainer.evalLoss(gpa, &model, &ad, s);
        ad.buf[best] = orig;
        const numeric = (lp - lm) / (2 * eps);
        try std.testing.expectApproxEqRel(numeric, grad[best], 0.05);
    }
}

const Events = struct {
    started: u32 = 0,
    finished: u32 = 0,
    failed: u32 = 0,
    wakeups: u32 = 0,
    deadline: ?i64 = null,
    fn wake(ctx: *anyopaque, d: ?i64) void {
        const e: *Events = @ptrCast(@alignCast(ctx));
        e.wakeups += 1;
        e.deadline = d;
    }
    fn on(ctx: *anyopaque, ev: schedule.Event) void {
        const e: *Events = @ptrCast(@alignCast(ctx));
        switch (ev) {
            .started => e.started += 1,
            .finished => e.finished += 1,
            .failed, .canceled => e.failed += 1,
            .progress => {},
        }
    }
};

test "gpu: home transcribes, learns from a fix, trains, and reports it" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const wav_bytes = Io.Dir.cwd().readFileAlloc(io, "testdata/hello.wav", gpa, .limited(16 << 20)) catch return error.SkipZigTest;
    defer gpa.free(wav_bytes);
    Io.Dir.cwd().access(io, model_path, .{}) catch return error.SkipZigTest;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);

    const h = home_mod.Home.open(gpa, io, .{
        .model_path = model_path,
        .profile_dir = dir,
        .schedule = .{ .window = .{ .start_minute = 0, .end_minute = 0 }, .min_new_samples = 1 }, // any time, 1 sample
        .training = .{ .epochs = 2 },
    }) catch |err| switch (err) {
        error.CudaUnavailable, error.CudaNoDevice => return error.SkipZigTest,
        else => return err,
    };
    defer h.close();
    var events: Events = .{};
    h.setListener(.{ .ctx = &events, .onEvent = Events.on });
    h.setWakeup(.{ .ctx = &events, .onWakeup = Events.wake });
    try std.testing.expectEqual(@as(?i64, null), events.deadline); // nothing pending

    const r = try h.transcribe(gpa, wav_bytes, "iphone");
    defer gpa.free(r.text);
    try std.testing.expect(r.id != 0);
    // A transcript waiting for feedback: Home wants a tick when silence expires (60 s).
    const now_ms: i64 = @intCast(@divFloor(Io.Clock.real.now(io).nanoseconds, std.time.ns_per_ms));
    try std.testing.expect(events.deadline != null and events.deadline.? > now_ms + 50_000);
    try std.testing.expect((try h.feedback(r.id, text, true)) == .learned);
    try std.testing.expectEqual(@as(u32, 1), h.trainingStatus().pending_samples);

    // The schedule is "any time, 1 sample": the fix makes a tick due now.
    try std.testing.expect(events.deadline.? <= now_ms + 1000);
    _ = h.tick();
    try std.testing.expectEqual(schedule.State.running, h.trainingStatus().state);
    try std.testing.expectEqual(@as(u32, 1), events.started);
    // The run finishes on its own thread: no more ticks needed.
    var waited: usize = 0;
    while (h.trainingStatus().state == .running and waited < 600) : (waited += 1) {
        io.sleep(.fromMilliseconds(100), .awake) catch {};
    }
    const st = h.trainingStatus();
    try std.testing.expectEqual(@as(?[]const u8, null), st.last_error);
    try std.testing.expectEqual(@as(u32, 1), events.finished);
    try std.testing.expectEqual(@as(u32, 1), st.version);
    try std.testing.expectEqual(@as(u32, 0), st.pending_samples);
    try std.testing.expect(st.last_loss != null);
    try std.testing.expectEqual(@as(?i64, null), h.tick()); // nothing left to do

    // Still transcribes with the new adapter merged in.
    const r2 = try h.transcribe(gpa, wav_bytes, "iphone");
    defer gpa.free(r2.text);
    try std.testing.expect(std.mem.indexOf(u8, r2.text, "Hello") != null);
}

// ------------------------------------------------------------------ stt.Local

const local_mod = @import("local.zig");
const sync = @import("sync.zig");
const training = @import("training.zig");
const Profile = @import("profile.zig").Profile;

fn openLocalCuda(dir: []const u8, svc: @import("service.zig").Options) !*local_mod.Local {
    return local_mod.Local.open(std.testing.allocator, std.testing.io, .{
        .model_path = model_path,
        .profile_dir = dir,
        .backend = .cuda,
        .service = svc,
    }) catch |err| switch (err) {
        error.CudaUnavailable, error.CudaNoDevice => return error.SkipZigTest,
        else => return err,
    };
}

test "gpu: Local (cuda backend): transcribe, fix, bundle home, adapter update, wakeups" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const wav_bytes = Io.Dir.cwd().readFileAlloc(io, "testdata/hello.wav", gpa, .limited(16 << 20)) catch return error.SkipZigTest;
    defer gpa.free(wav_bytes);
    Io.Dir.cwd().access(io, model_path, .{}) catch return error.SkipZigTest;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const ldir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/laptop", .{tmp.sub_path});
    defer gpa.free(ldir);
    const hdir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/home", .{tmp.sub_path});
    defer gpa.free(hdir);

    {
        const l = try openLocalCuda(ldir, .{});
        defer l.close();
        var events: Events = .{};
        l.setWakeup(.{ .ctx = &events, .onWakeup = Events.wake });
        try std.testing.expectEqual(@as(u32, 1), events.wakeups);
        try std.testing.expectEqual(@as(?i64, null), events.deadline);
        try std.testing.expectEqual(@as(?i64, null), l.tick()); // idle

        // Transcribe; the silence auto-confirm becomes the deadline (60 s).
        const r = try l.transcribe(gpa, wav_bytes);
        defer gpa.free(r.text);
        try std.testing.expect(r.id != 0);
        try std.testing.expect(std.mem.indexOf(u8, r.text, "Hello") != null);
        const now_ms: i64 = @intCast(@divFloor(Io.Clock.real.now(io).nanoseconds, std.time.ns_per_ms));
        try std.testing.expect(events.deadline != null and events.deadline.? > now_ms + 50_000);
        try std.testing.expect(l.tick() != null); // still waiting for feedback

        // A fix: learned, nothing pending any more.
        const out = try l.feedback(r.id, text, true);
        try std.testing.expect(out == .learned);
        try std.testing.expectEqual(@as(?i64, null), events.deadline);
        try std.testing.expectEqual(@as(?i64, null), l.tick());
        try std.testing.expectEqual(@as(u32, 1), l.status().pending_export);

        // Device -> home: the bundle imports into a home profile.
        var b = (try l.exportBundle(gpa, sync.default_bundle_bytes)).?;
        defer b.deinit(gpa);
        var hp = try Profile.load(gpa, io, hdir);
        defer hp.deinit();
        const rep = try sync.importBundle(gpa, &hp, b.bytes, "laptop");
        try std.testing.expectEqual(@as(u32, 1), rep.fixes);
        try l.commitExport(b.through);
        try std.testing.expectEqual(@as(u32, 0), l.status().pending_export);
        try std.testing.expect((try l.exportBundle(gpa, sync.default_bundle_bytes)) == null);

        // Home trains v1; the laptop imports it and switches the engine to it.
        var job = try training.collect(gpa, &hp);
        defer job.deinit();
        _ = try training.run(gpa, io, &l.model, &job, .{ .epochs = 2 }, null, null);
        try training.commit(&hp, &job);
        const up = (try sync.exportUpdate(gpa, &hp, l.status().profile_version)).?;
        defer gpa.free(up);
        try std.testing.expectEqual(@as(u32, 0), l.adapterVersion());
        try std.testing.expectEqual(@as(u32, 1), try l.importUpdate(gpa, up));
        try std.testing.expectEqual(@as(u32, 1), l.adapterVersion());
        try std.testing.expectEqual(@as(u32, 1), try l.importUpdate(gpa, up)); // idempotent
        try std.testing.expectError(error.BadUpdate, l.importUpdate(gpa, "STTBUN01 not an update"));

        // Still transcribes with the new adapter (and the learned rule).
        const r2 = try l.transcribe(gpa, wav_bytes);
        defer gpa.free(r2.text);
        try std.testing.expect(std.mem.indexOf(u8, r2.text, "Hello") != null);
        try std.testing.expectError(error.BadAudio, l.transcribe(gpa, "not audio"));
    }

    // Reopening picks up the imported adapter; silence confirms at the deadline.
    {
        const l = try openLocalCuda(ldir, .{ .confirm_after_ms = 200 });
        defer l.close();
        try std.testing.expectEqual(@as(u32, 1), l.adapterVersion());
        const pending0 = l.status().pending_export;
        const r = try l.transcribe(gpa, wav_bytes);
        defer gpa.free(r.text);
        const d = l.tick().?;
        const now_ms: i64 = @intCast(@divFloor(Io.Clock.real.now(io).nanoseconds, std.time.ns_per_ms));
        try std.testing.expect(d <= now_ms + 200);
        io.sleep(.fromMilliseconds(d - now_ms + 50), .awake) catch {};
        try std.testing.expectEqual(@as(?i64, null), l.tick()); // confirmed by silence
        try std.testing.expectEqual(pending0 + 1, l.status().pending_export);
    }
}

test "Local: the NPU backend is unavailable in builds without the NPU engine" {
    const gpa = std.testing.allocator;
    const npu = @import("npu/engine.zig");
    if (!npu.enabled) {
        // The stub never touches the model, so this needs no model file.
        var no_model: Model = undefined;
        try std.testing.expectError(error.NpuUnavailable, npu.Engine.init(gpa, &no_model, .{}));
    }

    // Through Local (needs the model).
    Io.Dir.cwd().access(std.testing.io, model_path, .{}) catch return error.SkipZigTest;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer gpa.free(dir);
    const cfg: local_mod.Config = .{ .model_path = model_path, .profile_dir = dir };
    if (!npu.enabled) {
        try std.testing.expectError(error.NpuUnavailable, local_mod.Local.open(gpa, std.testing.io, cfg));
        return;
    }
    // A build with the NPU engine: opening needs the QNN runtime and an NPU.
    const l = local_mod.Local.open(gpa, std.testing.io, cfg) catch |err| {
        std.debug.print("Local NPU open: {t} (skipping)\n", .{err});
        return error.SkipZigTest;
    };
    l.close();
}
