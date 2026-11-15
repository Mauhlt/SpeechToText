//! coreml-dev: development checks of the Core ML engine. A separate build
//! so the library's build.zig stays small (and so that only this build, or
//! an iOS / macOS target, ever fetches TranslateCoreML):
//!
//!   cd src/coreml/dev
//!   zig build                        -> zig-out/bin/coreml-dev (this host;
//!                                       --runtime reference off Apple)
//!   zig build -Dtarget=aarch64-macos -> a Mac binary (Core ML is opened with
//!                                       dlopen, so this cross-links from Windows)
//!   zig build run -- [args]          (runs from the SpeechToText folder)
//!   zig build apple-check            stt with the Core ML backend for
//!                                    aarch64-ios and aarch64-macos (objects
//!                                    only, no SDK)
//!   zig build ios-check              SttCoreML.app's code for aarch64-ios
//!                                    (object only, no SDK)
//!   zig build ios                    (Mac) zig-out/ios/SttCoreML.app, with
//!                                    the model (-Dios-model, default
//!                                    models/ggml-tiny.en.bin) and
//!                                    the speech sample (-Dios-wav, default
//!                                    testdata/speech.wav, git-ignored) inside
//!   zig build ios-sign               sign it (PORTAL_SIGN_IDENTITY,
//!                                    PORTAL_PROVISIONING_PROFILE, PORTAL_TEAM_ID)
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const exe = b.addExecutable(.{
        .name = "coreml-dev",
        .root_module = b.createModule(.{
            .root_source_file = b.path("main.zig"),
            .target = target,
            .optimize = .fast,
            .imports = &.{.{ .name = "stt", .module = sttModule(b, target, .fast) }},
        }),
    });
    b.installArtifact(exe);
    const run = b.addRunArtifact(exe);
    run.setCwd(b.path("../../.."));
    run.addPassthruArgs();
    b.step("run", "Run coreml-dev from the SpeechToText folder").dependOn(&run.step);

    // The library with the Core ML backend for the Apple targets: every
    // engine method and Transcriber with .coreml analysed (apple_check.zig).
    const apple_check = b.step("apple-check", "Compile stt with the Core ML backend for aarch64-ios and aarch64-macos (objects only)");
    for ([_]struct { name: []const u8, t: std.Build.ResolvedTarget }{
        .{ .name = "stt-ios-check", .t = iosTarget(b) },
        .{ .name = "stt-macos-check", .t = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .macos, .os_version_min = .{ .semver = .{ .major = 14, .minor = 0, .patch = 0 } } }) },
    }) |c| {
        const root = b.createModule(.{
            .root_source_file = b.path("apple_check.zig"),
            .target = c.t,
            .optimize = .debug,
            .link_libc = true,
            .imports = &.{.{ .name = "stt", .module = sttModule(b, c.t, .debug) }},
        });
        apple_check.dependOn(&b.addObject(.{ .name = c.name, .root_module = root }).step);
    }

    addIos(b);
}

/// stt with the Core ML backend (on for every target here, so a host build
/// runs it on the reference interpreter).
fn sttModule(b: *std.Build, t: std.Build.ResolvedTarget, optimize: std.lang.Optimize) *std.Build.Module {
    return b.dependency("stt", .{ .target = t, .optimize = optimize, .coreml = true }).module("stt");
}

fn iosTarget(b: *std.Build) std.Build.ResolvedTarget {
    return b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .ios, .os_version_min = .{ .semver = .{ .major = 17, .minor = 0, .patch = 0 } } });
}

fn addIos(b: *std.Build) void {
    const bundle_id = b.option([]const u8, "ios-bundle-id", "SttCoreML.app bundle id (default local.stt.coreml; must match the provisioning profile)") orelse "local.stt.coreml";
    const model: std.Build.LazyPath = if (b.option([]const u8, "ios-model", "ggml model file to bundle (default models/ggml-tiny.en.bin of SpeechToText)")) |p| .{ .cwd_relative = p } else b.path("../../../models/ggml-tiny.en.bin");
    // The speech sample the app transcribes: a recording of your own (the repo
    // ships no audio; THIRD_PARTY.md, Test data).
    const speech: std.Build.LazyPath = if (b.option([]const u8, "ios-wav", "speech sample to bundle (default testdata/speech.wav of SpeechToText, git-ignored)")) |p| .{ .cwd_relative = p } else b.path("../../../testdata/speech.wav");
    const env = &b.graph.environ_map;
    const sdk = b.option([]const u8, "ios-sdk", "iPhoneOS SDK path (env IOS_SDK; default: xcrun --sdk iphoneos --show-sdk-path on a Mac)") orelse env.get("IOS_SDK") orelse if (b.graph.host.result.os.tag == .macos)
        std.mem.trim(u8, b.run(&.{ "xcrun", "--sdk", "iphoneos", "--show-sdk-path" }), " \r\n")
    else
        null;
    const t = iosTarget(b);
    // TranslateCoreML's signing tool, through stt's (lazy) dependency so the
    // pin lives in one place: null until fetched, then build() runs again.
    const tools = b.dependency("stt", .{}).builder.lazyDependency("TranslateCoreML", .{}) orelse return;

    const check = b.addObject(.{ .name = "sttcoreml-ios-check", .root_module = appModule(b, t, .debug) });
    b.step("ios-check", "Compile SttCoreML.app's code for aarch64-ios (object only)").dependOn(&check.step);

    const app_step = b.step("ios", "On a Mac: build zig-out/ios/SttCoreML.app");
    const sign_step = b.step("ios-sign", "Sign zig-out/ios/SttCoreML.app (env PORTAL_SIGN_IDENTITY, PORTAL_PROVISIONING_PROFILE, PORTAL_TEAM_ID)");
    const sdk_path = sdk orelse {
        const fail = b.addFail("zig build ios needs the iPhoneOS SDK: run it on a Mac with Xcode, or pass -Dios-sdk=<path> / IOS_SDK");
        app_step.dependOn(&fail.step);
        sign_step.dependOn(&fail.step);
        return;
    };
    const mod = appModule(b, t, .fast);
    mod.addSystemFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "System/Library/Frameworks" }) });
    mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "usr/lib" }) });
    mod.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "usr/include" }) });
    for ([_][]const u8{ "UIKit", "Foundation" }) |fw| mod.linkFramework(fw, .{});
    mod.linkSystemLibrary("objc", .{});
    const exe = b.addExecutable(.{ .name = "SttCoreML", .root_module = mod });
    exe.setLibCFile(b.addWriteFiles().add("libc-ios.txt", b.fmt(
        \\include_dir={s}/usr/include
        \\sys_include_dir={s}/usr/include
        \\crt_dir={s}/usr/lib
        \\msvc_lib_dir=
        \\kernel32_lib_dir=
        \\gcc_dir=
        \\
    , .{ sdk_path, sdk_path, sdk_path })));

    const app = b.addWriteFiles();
    _ = app.addCopyFile(exe.getEmittedBin(), "SttCoreML.app/SttCoreML");
    _ = app.add("SttCoreML.app/Info.plist", infoPlist(b, bundle_id));
    _ = app.add("SttCoreML.app/PkgInfo", "APPL????");
    _ = app.addCopyFile(model, "SttCoreML.app/stt/model.bin");
    _ = app.addCopyFile(speech, "SttCoreML.app/stt/speech.wav");
    const install = b.addInstallDirectory(.{ .source_dir = app.getDirectory().path(b, "SttCoreML.app"), .install_dir = .prefix, .install_subdir = "ios/SttCoreML.app" });
    app_step.dependOn(&install.step);

    const sign = b.addRunArtifact(tools.artifact("ios-sign"));
    sign.addDirectoryArg(app.getDirectory().path(b, "SttCoreML.app"));
    const signed = sign.addOutputDirectoryArg("signed");
    sign.addArg(bundle_id);
    sign.has_side_effects = true;
    const install_signed = b.addInstallDirectory(.{ .source_dir = signed.path(b, "SttCoreML.app"), .install_dir = .prefix, .install_subdir = "ios/SttCoreML.app" });
    install_signed.step.dependOn(&install.step);
    sign_step.dependOn(&install_signed.step);
}

fn appModule(b: *std.Build, t: std.Build.ResolvedTarget, optimize: std.lang.Optimize) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = b.path("ios_main.zig"),
        .target = t,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "stt", .module = sttModule(b, t, optimize) }},
    });
}

/// Info.plist (configuration text).
fn infoPlist(b: *std.Build, bundle_id: []const u8) []const u8 {
    return b.fmt(
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        \\<plist version="1.0">
        \\<dict>
        \\    <key>CFBundleDevelopmentRegion</key><string>en</string>
        \\    <key>CFBundleExecutable</key><string>SttCoreML</string>
        \\    <key>CFBundleIdentifier</key><string>{s}</string>
        \\    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
        \\    <key>CFBundleName</key><string>SttCoreML</string>
        \\    <key>CFBundleDisplayName</key><string>stt Core ML</string>
        \\    <key>CFBundlePackageType</key><string>APPL</string>
        \\    <key>CFBundleShortVersionString</key><string>0.1.0</string>
        \\    <key>CFBundleVersion</key><string>1</string>
        \\    <key>CFBundleSupportedPlatforms</key><array><string>iPhoneOS</string></array>
        \\    <key>DTPlatformName</key><string>iphoneos</string>
        \\    <key>MinimumOSVersion</key><string>17.0</string>
        \\    <key>LSRequiresIPhoneOS</key><true/>
        \\    <key>UIDeviceFamily</key><array><integer>1</integer></array>
        \\    <key>UIRequiredDeviceCapabilities</key><array><string>arm64</string></array>
        \\    <key>UILaunchScreen</key><dict/>
        \\    <key>UIFileSharingEnabled</key><true/>
        \\    <key>LSSupportsOpeningDocumentsInPlace</key><true/>
        \\</dict>
        \\</plist>
        \\
    , .{bundle_id});
}
