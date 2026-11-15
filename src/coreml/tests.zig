//! Core ML backend checks that run on any OS (built for iOS / macOS targets,
//! elsewhere with `zig build test -Dcoreml`). They need
//! models/ggml-tiny.en.bin and testdata/hello.wav (skipped otherwise); the
//! comparisons with the GPU also need an NVIDIA GPU (skipped otherwise).
const std = @import("std");
const Io = std.Io;
const Model = @import("../model.zig").Model;
const audio = @import("../audio.zig");
const Transcriber = @import("../transcriber.zig").Transcriber;
const gpu = @import("../gpu/engine.zig");
const Adapter = @import("../adapter.zig").Adapter;
const engine = @import("engine.zig");
const net = @import("net.zig");
const frontend = @import("frontend.zig");
const coreml = @import("coreml");

const model_path = "models/ggml-tiny.en.bin";

fn loadModel() !Model {
    return Model.load(std.testing.allocator, std.testing.io, model_path) catch error.SkipZigTest;
}
fn loadAudio() ![]f32 {
    return audio.load(std.testing.allocator, std.testing.io, "testdata/hello.wav", .{}) catch error.SkipZigTest;
}

test "encoder and decoder programs serialize and read back" {
    const gpa = std.testing.allocator;
    var model = try loadModel();
    defer model.deinit();
    const hp = model.hp;
    {
        var b = try coreml.mil.Builder.init(gpa, .{ .target = .ios17 });
        defer b.deinit();
        try net.encoder(&b, &model, .{});
        var f = try coreml.package.build(gpa, &b);
        defer f.deinit(gpa);
        var s = try coreml.decode.decode(gpa, f.model);
        defer s.deinit();
        try std.testing.expectEqual(@as(usize, 1), s.inputs.len);
        try std.testing.expectEqualStrings("mel", s.inputs[0].name);
        try std.testing.expectEqual(2 * hp.n_text_layer, s.outputs.len);
        try std.testing.expectEqualStrings("ck0", s.outputs[0].name);
        try std.testing.expectEqualSlices(i64, &.{ 1, @intCast(hp.n_text_head), 1500, 64 }, s.outputs[0].shape);
        try std.testing.expectEqual(b.ops.items.len, s.ops.len);
        // every weight matrix went to the blob file
        try std.testing.expect(try coreml.blob.count(f.weights) > 10 * hp.n_audio_layer);
    }
    {
        const slots = try net.baseWeights(gpa, &model);
        defer gpa.free(slots);
        var b = try coreml.mil.Builder.init(gpa, .{ .target = .ios17 });
        defer b.deinit();
        try net.decoder(&b, &model, .{}, .{ .w = slots });
        var f = try coreml.package.build(gpa, &b);
        defer f.deinit(gpa);
        var s = try coreml.decode.decode(gpa, f.model);
        defer s.deinit();
        try std.testing.expectEqual(2 + 4 * hp.n_text_layer, s.inputs.len);
        try std.testing.expectEqual(2 * hp.n_text_layer + 1, s.outputs.len);
        try std.testing.expectEqualStrings("logits", s.outputs[s.outputs.len - 1].name);
        for (s.inputs) |t| try std.testing.expectEqual(@as(i32, 65552), t.dtype); // all fp16
    }
}

/// The GPU engine, or skip.
fn gpuEngine(gpa: std.mem.Allocator, model: *const Model) !*gpu.Engine {
    return gpu.Engine.init(gpa, model) catch error.SkipZigTest;
}

test "host log-mel matches the GPU's" {
    const gpa = std.testing.allocator;
    var model = try loadModel();
    defer model.deinit();
    const samples = try loadAudio();
    defer gpa.free(samples);
    const g = try gpuEngine(gpa, &model);
    defer g.deinit(gpa);
    try g.encode(samples);
    try g.gpu.sync();
    const n = model.hp.n_mels * 3000;
    const want = try gpa.alloc(f32, n);
    defer gpa.free(want);
    try g.gpu.download(std.mem.sliceAsBytes(want), g.b.spec);
    try g.gpu.sync();
    const got = try gpa.alloc(f32, n);
    defer gpa.free(got);
    frontend.logMel(&model, samples, got);
    var max_err: f32 = 0;
    for (got, want) |a, b| max_err = @max(max_err, @abs(a - b));
    std.debug.print("coreml frontend: log-mel max abs err vs GPU {d:.6}\n", .{max_err});
    try std.testing.expect(max_err < 2e-3);
}

test "Core ML programs on the reference interpreter transcribe like the GPU" {
    const gpa = std.testing.allocator;
    var model = try loadModel();
    defer model.deinit();
    const samples = try loadAudio();
    defer gpa.free(samples);

    var gtr = Transcriber.init(gpa, &model, .{}) catch return error.SkipZigTest;
    const want = try gtr.transcribe(gpa, samples);
    defer gpa.free(want);
    gtr.deinit();

    var tr = try Transcriber.init(gpa, &model, .{ .backend = .{ .coreml = .{ .runtime = .reference } } });
    defer tr.deinit();
    const got = try tr.transcribe(gpa, samples);
    defer gpa.free(got);
    std.debug.print("coreml (reference, fp16 emulated): \"{s}\"\nGPU:                               \"{s}\"\n", .{ got, want });
    try std.testing.expectEqualStrings(want, got);
}
