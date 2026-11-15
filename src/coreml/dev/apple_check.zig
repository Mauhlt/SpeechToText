//! Root of `zig build apple-check`: references the transcriber with the
//! Core ML backend and every Core ML engine method, so an aarch64-ios /
//! aarch64-macos compile analyses all of it (objects only; nothing runs).
const std = @import("std");
const stt = @import("stt");
const E = stt.coreml.engine.Engine;

comptime {
    _ = &E.init;
    _ = &E.deinit;
    _ = &E.encode;
    _ = &E.step;
    _ = &E.pick;
    _ = &E.setAdapter;
    _ = &E.deviceName;
    _ = &E.crossKV;
    _ = &E.melSpectrogram;
    _ = &stt.Transcriber.init;
    _ = &stt.Transcriber.transcribe;
    _ = &stt.Transcriber.deinit;
    _ = &check;
}

/// What an iPhone app would call (never run here).
pub export fn stt_coreml_apple_check(model: *const stt.Model, cache: [*:0]const u8) c_int {
    return check(model, std.mem.span(cache)) catch 1;
}

fn check(model: *const stt.Model, cache: []const u8) !c_int {
    const gpa = std.heap.c_allocator;
    var tr = try stt.Transcriber.init(gpa, model, .{ .backend = .{ .coreml = .{ .cache_dir = cache } } });
    defer tr.deinit();
    const text = try tr.transcribe(gpa, &.{});
    gpa.free(text);
    return 0;
}
