const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    // Neural-net inference is ~100x slower in Debug, so default to fast.
    const optimize = b.option(std.lang.Optimize, "optimize", "Optimization mode (default: fast)") orelse .fast;

    // The library module other Zig projects import: `@import("stt")`.
    const mod = b.addModule("stt", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "Audio", .module = b.dependency("Audio", .{ .target = target }).module("Audio") },
            // QNN bindings for the laptop NPU path (stt.Local). Building needs
            // the QAIRT SDK headers (see ../TranslateNPU); nothing from the SDK ships.
            .{ .name = "qnn", .module = b.dependency("TranslateNPU", .{ .target = target, .optimize = optimize }).module("qnn") },
            // Core ML backend (coreml/): MIL writer + reference interpreter on every
            // OS, Core ML runtime bindings on iOS / macOS (dlopen'd, no link-time frameworks).
            .{ .name = "coreml", .module = b.dependency("TranslateCoreML", .{ .target = target, .optimize = optimize }).module("coreml") },
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

    // `zig build apple-check`: compile the library with the Core ML backend
    // for aarch64-ios and aarch64-macos (objects only, no Apple SDK; part of
    // `zig build test`). iOS uses TranslateCoreML's patched copy of the Zig
    // lib dir (std.Io.Threaded does not compile for iOS in this Zig).
    const apple_check = b.step("apple-check", "Compile stt with the Core ML backend for aarch64-ios and aarch64-macos (objects only)");
    {
        const coreml_tools = b.dependency("TranslateCoreML", .{});
        const lib_run = b.addRunArtifact(coreml_tools.artifact("ios-zig-lib"));
        lib_run.addDirectoryArg(.zig_lib);
        const ios_lib = lib_run.addOutputDirectoryArg("zig-lib");
        const checks = [_]struct { name: []const u8, os: std.Target.Os.Tag, min: u32, lib: ?std.Build.LazyPath }{
            .{ .name = "stt-ios-check", .os = .ios, .min = 17, .lib = ios_lib },
            .{ .name = "stt-macos-check", .os = .macos, .min = 14, .lib = null },
        };
        for (checks) |c| {
            const t = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = c.os, .os_version_min = .{ .semver = .{ .major = c.min, .minor = 0, .patch = 0 } } });
            const m = b.createModule(.{
                .root_source_file = b.path("src/root.zig"),
                .target = t,
                .optimize = .Debug,
                .link_libc = true,
                .imports = &.{
                    .{ .name = "Audio", .module = b.dependency("Audio", .{ .target = t }).module("Audio") },
                    .{ .name = "qnn", .module = b.dependency("TranslateNPU", .{ .target = t, .optimize = .Debug }).module("qnn") },
                    .{ .name = "coreml", .module = b.dependency("TranslateCoreML", .{ .target = t, .optimize = .Debug }).module("coreml") },
                },
            });
            m.addAnonymousImport("stt_ptx", .{ .root_source_file = wf.getDirectory().path(b, "ptx.zig") });
            const root = b.createModule(.{
                .root_source_file = b.path("src/coreml/apple_check.zig"),
                .target = t,
                .optimize = .Debug,
                .link_libc = true,
                .imports = &.{.{ .name = "stt", .module = m }},
            });
            const obj = b.addObject(.{ .name = c.name, .root_module = root, .zig_lib_dir = c.lib });
            apple_check.dependOn(&obj.step);
        }
    }
    test_step.dependOn(apple_check);
}
