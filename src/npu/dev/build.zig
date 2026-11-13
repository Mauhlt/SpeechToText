//! npu-dev: development check of the NPU engine against the GPU engine.
//! A separate build so the library's build.zig stays untouched:
//!
//!   cd src\npu\dev
//!   zig build                       -> zig-out\bin\npu-dev.exe
//!   zig build run -- --backend htp  (runs from the SpeechToText folder)
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const stt = b.dependency("stt", .{ .target = target });
    const exe = b.addExecutable(.{
        .name = "npu-dev",
        .root_module = b.createModule(.{
            .root_source_file = b.path("main.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .imports = &.{.{ .name = "stt", .module = stt.module("stt") }},
        }),
    });
    b.installArtifact(exe);
    const run = b.addRunArtifact(exe);
    run.setCwd(b.path("../../.."));
    run.addPassthruArgs();
    b.step("run", "Run npu-dev from the SpeechToText folder").dependOn(&run.step);
}
