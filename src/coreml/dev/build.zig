//! coreml-dev: development check of the Core ML engine against the GPU
//! engine. A separate build so the library's build.zig stays small:
//!
//!   cd src/coreml/dev
//!   zig build                        -> zig-out/bin/coreml-dev (this host;
//!                                       --runtime reference)
//!   zig build -Dtarget=aarch64-macos -> a Mac binary (Core ML is dlopen'd,
//!                                       so this cross-links from Windows)
//!   zig build run -- [args]          (runs from the SpeechToText folder)
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const stt = b.dependency("stt", .{ .target = target });
    const exe = b.addExecutable(.{
        .name = "coreml-dev",
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
    b.step("run", "Run coreml-dev from the SpeechToText folder").dependOn(&run.step);
}
