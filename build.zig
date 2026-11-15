const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    // Neural-net inference is ~100x slower in Debug, so default to fast.
    const optimize = b.option(std.lang.Optimize, "optimize", "Optimization mode (default: fast)") orelse .fast;

    // The NPU engine (src/npu: stt.Local on a Snapdragon laptop's Hexagon
    // NPU, through QNN) is built for aarch64-windows only. Every other
    // target gets an engine that reports error.NpuUnavailable, and never
    // compiles QNN or needs the QAIRT SDK. -Dnpu=true turns it on elsewhere
    // (QNN's CPU backend on x86_64-windows), with the SDK from -Dsdk or
    // QNN_SDK_ROOT (TranslateNPU fetches the SDK for aarch64-windows only).
    const npu_default = target.result.os.tag == .windows and target.result.cpu.arch == .aarch64;
    const npu = b.option(bool, "npu", "Build the NPU engine (QNN; default: only for aarch64-windows)") orelse npu_default;
    const sdk = b.option([]const u8, "sdk", "QAIRT/QNN SDK root for the NPU engine (passed on to TranslateNPU)");
    const stt_options = b.addOptions();
    // Core ML backend (src/coreml): on for iOS / macOS targets. Off elsewhere,
    // so TranslateCoreML (a lazy dependency) is never fetched or compiled;
    // -Dcoreml turns it on for checks on its reference interpreter
    // (`zig build test -Dcoreml` compares the programs with the GPU).
    const coreml = b.option(bool, "coreml", "Include the Core ML backend (default: only for iOS / macOS targets)") orelse isApple(target);
    stt_options.addOption(bool, "npu", npu);
    stt_options.addOption(bool, "coreml", coreml);

    // The library module other Zig projects import: `@import("stt")`.
    const mod = b.addModule("stt", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "Audio", .module = b.dependency("Audio", .{ .target = target }).module("Audio") },
            .{ .name = "stt_options", .module = stt_options.createModule() },
        },
    });

    // GPU kernels: src/gpu/kernels.zig compiled by Zig's NVPTX backend to PTX
    // text, which is embedded in the library and JIT-compiled by the NVIDIA
    // driver for whichever GPU is installed (Pascal / GTX 10xx and newer).
    // The library needs an NVIDIA GPU at runtime (Windows or Linux).
    const wf = b.addWriteFiles();
    {
        const kernels = b.addObject(.{
            .name = "stt_kernels",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/gpu/kernels.zig"),
                .target = b.resolveTargetQuery(.{
                    .cpu_arch = .nvptx64,
                    .os_tag = .cuda,
                    .cpu_model = .{ .explicit = &std.Target.nvptx.cpu.sm_60 },
                    .cpu_features_add = std.Target.nvptx.featureSet(&.{.ptx63}),
                }),
                .optimize = .fast,
            }),
        });
        _ = wf.addCopyFile(kernels.getEmittedAsm(), "kernels.ptx");
        _ = wf.add("ptx.zig",
            \\pub const enabled = true;
            \\pub const source: [:0]const u8 = @embedFile("kernels.ptx");
            \\
        );
    }
    mod.addAnonymousImport("stt_ptx", .{ .root_source_file = wf.getDirectory().path(b, "ptx.zig") });
    if (coreml) {
        // MIL writer + reference interpreter on every OS, Core ML runtime
        // bindings on iOS / macOS (dlopen'd, no link-time frameworks).
        if (b.lazyDependency("TranslateCoreML", .{ .target = target, .optimize = optimize })) |dep| mod.addImport("coreml", dep.module("coreml"));
    }

    const exe = b.addExecutable(.{
        .name = "stt",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "stt", .module = mod }},
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    b.step("run", "Run the stt CLI").dependOn(&run_cmd.step);

    // `stt-local`: the laptop program Portal runs as a child process
    // (stt.Local on the NPU behind a JSON-lines protocol on stdin/stdout).
    // Target: aarch64-windows (Snapdragon X); it also builds for the host,
    // where `--backend cuda` runs the same flow on an NVIDIA GPU.
    const local_exe = b.addExecutable(.{
        .name = "stt-local",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/local_main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "stt", .module = mod }},
        }),
    });
    b.installArtifact(local_exe);

    const test_step = b.step("test", "Run tests");
    const mod_tests = b.addTest(.{ .root_module = mod });
    test_step.dependOn(&b.addRunArtifact(mod_tests).step);

    // stt-local's own unit tests, and an end-to-end protocol test that
    // spawns the built executable.
    const local_unit = b.addTest(.{ .root_module = local_exe.root_module });
    test_step.dependOn(&b.addRunArtifact(local_unit).step);
    const proto_opts = b.addOptions();
    proto_opts.addOptionPath("exe_path", local_exe.getEmittedBin());
    const proto_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/local_protocol_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "stt_local_build", .module = proto_opts.createModule() }},
        }),
    });
    test_step.dependOn(&b.addRunArtifact(proto_tests).step);

    // QNN bindings for the NPU engine: TranslateNPU, a lazy dependency
    // fetched only when one of these steps (the ones compiling the "stt"
    // module here) runs with the NPU engine on. A package that compiles
    // "stt" with -Dnpu itself must use one of them (Portal builds
    // stt-local). Nothing from the SDK ships: the laptop loads the QNN
    // runtime installed on it.
    if (npu) qnnImport(b, mod, &.{ &local_exe.step, &exe.step, &mod_tests.step, &local_unit.step }, .{
        .target = target,
        .optimize = optimize,
        .sdk = sdk,
    });
}

fn qnnImport(b: *std.Build, mod: *std.Build.Module, steps: []const *std.Build.Step, args: anytype) void {
    if (b.dependencyStep(steps[0], "TranslateNPU", args)) |dep| {
        mod.addImport("qnn", dep.module("qnn"));
        return;
    }
    for (steps[1..]) |s| b.declareDependencyStep(s, "TranslateNPU");
}

fn isApple(t: std.Build.ResolvedTarget) bool {
    return switch (t.result.os.tag) {
        .ios, .macos => true,
        else => false,
    };
}
