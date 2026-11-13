//! `stt` command-line tool, for testing on the home desktop. Transcripts go
//! to stdout; status messages go to stderr. Real use goes through Portal,
//! which embeds the library (`stt.Home`).
const std = @import("std");
const Io = std.Io;
const stt = @import("stt");

const usage =
    \\stt - speech to text in pure Zig, on an NVIDIA GPU
    \\
    \\USAGE
    \\  stt transcribe <audio>          Transcribe a WAV / MP3 / QOA file
    \\  stt bench <audio> [--runs n]    Time transcription and show where GPU time goes
    \\  stt train                       Train the personal adapter on the GPU now
    \\  stt status                      Samples waiting, adapter version, training schedule
    \\
    \\OPTIONS
    \\  --model <path>    Model file (default models/ggml-base.en.bin)
    \\  --prompt <text>   Vocabulary hint, e.g. "Zig, comptime, allocator"
    \\  --profile <dir>   Learning profile folder (default: %LOCALAPPDATA%\\stt\\profile,
    \\                    or ~/.local/share/stt/profile)
    \\  --no-profile      Don't use the learning profile (base model, no rules)
    \\  --epochs <n>      train: passes over your samples (default 3)
    \\  --lr <x>          train: learning rate (default: muon 0.004, adam 0.001)
    \\  --optimizer <o>   train: muon (default) or adam
    \\
;

const Args = struct {
    cmd: []const u8 = "",
    positional: ?[]const u8 = null,
    model: []const u8 = "models/ggml-base.en.bin",
    prompt: []const u8 = "",
    /// "" = use the default location (see `defaultProfileDir`).
    profile: ?[]const u8 = "",
    epochs: usize = 3,
    lr: ?f32 = null,
    optimizer: stt.gpu.trainer.Optimizer = .muon,
    runs: usize = 3,
};

fn parseArgs(argv: []const [:0]const u8) !Args {
    var a: Args = .{};
    if (argv.len < 2) return error.Usage;
    a.cmd = argv[1];
    var i: usize = 2;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        const val = struct {
            fn get(v: []const [:0]const u8, idx: *usize) ![]const u8 {
                idx.* += 1;
                if (idx.* >= v.len) return error.Usage;
                return v[idx.*];
            }
        }.get;
        if (std.mem.eql(u8, arg, "--model")) {
            a.model = try val(argv, &i);
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            a.prompt = try val(argv, &i);
        } else if (std.mem.eql(u8, arg, "--profile")) {
            a.profile = try val(argv, &i);
        } else if (std.mem.eql(u8, arg, "--no-profile")) {
            a.profile = null;
        } else if (std.mem.eql(u8, arg, "--epochs")) {
            a.epochs = try std.fmt.parseInt(usize, try val(argv, &i), 10);
        } else if (std.mem.eql(u8, arg, "--optimizer")) {
            a.optimizer = std.meta.stringToEnum(stt.gpu.trainer.Optimizer, try val(argv, &i)) orelse return error.Usage;
        } else if (std.mem.eql(u8, arg, "--lr")) {
            a.lr = try std.fmt.parseFloat(f32, try val(argv, &i));
        } else if (std.mem.eql(u8, arg, "--runs")) {
            a.runs = @max(1, try std.fmt.parseInt(usize, try val(argv, &i), 10));
        } else if (std.mem.startsWith(u8, arg, "--")) {
            std.debug.print("unknown option {s}\n", .{arg});
            return error.Usage;
        } else if (a.positional == null) {
            a.positional = arg;
        } else return error.Usage;
    }
    return a;
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const argv = try init.minimal.args.toSlice(arena);

    var args = parseArgs(argv) catch usageExit();
    if (args.profile) |dir| if (dir.len == 0) {
        args.profile = try defaultProfileDir(arena, init.environ_map);
    };

    if (std.mem.eql(u8, args.cmd, "transcribe")) return cmdTranscribe(gpa, io, args);
    if (std.mem.eql(u8, args.cmd, "bench")) return cmdBench(gpa, io, args);
    if (std.mem.eql(u8, args.cmd, "train")) return cmdTrain(gpa, io, args);
    if (std.mem.eql(u8, args.cmd, "status")) return cmdStatus(gpa, io, args);
    usageExit();
}

fn loadModel(gpa: std.mem.Allocator, io: Io, path: []const u8) stt.Model {
    return stt.Model.load(gpa, io, path) catch |err| {
        std.debug.print("error: could not load model '{s}': {t}\n", .{ path, err });
        std.debug.print("see README.md (Setup) for where to get a model file\n", .{});
        std.process.exit(1);
    };
}

fn openTranscriber(gpa: std.mem.Allocator, model: *const stt.Model) stt.Transcriber {
    return stt.Transcriber.init(gpa, model, .{}) catch |err| {
        std.debug.print("error: cannot use the GPU ({t}); stt needs an NVIDIA GPU and driver\n", .{err});
        std.process.exit(1);
    };
}

fn loadAudio(gpa: std.mem.Allocator, io: Io, args: Args) []f32 {
    const path = args.positional orelse usageExit();
    return stt.audio.load(gpa, io, path, .{ .max_seconds = 4 * 3600 }) catch |err| {
        std.debug.print("error: cannot read audio '{s}': {t}\n", .{ path, err });
        std.process.exit(1);
    };
}

fn cmdTranscribe(gpa: std.mem.Allocator, io: Io, args: Args) !void {
    const samples = loadAudio(gpa, io, args);
    defer gpa.free(samples);
    var model = loadModel(gpa, io, args.model);
    defer model.deinit();
    var tr = openTranscriber(gpa, &model);
    defer tr.deinit();

    var profile: ?stt.Profile = null;
    defer if (profile) |*p| p.deinit();
    if (args.profile) |dir| profile = try stt.Profile.load(gpa, io, dir);
    if (profile) |*p| {
        if (try stt.training.loadAdapter(gpa, p, &model)) |ad_| {
            var ad = ad_;
            defer ad.deinit(gpa);
            try tr.setAdapter(&ad, p.version);
            std.debug.print("[profile] personal adapter v{d}\n", .{p.version});
        }
        const pt = try p.promptText(gpa, args.prompt);
        defer gpa.free(pt);
        try tr.setPrompt(pt);
    } else try tr.setPrompt(args.prompt);

    const raw = try tr.transcribe(gpa, samples);
    defer gpa.free(raw);
    const text = if (profile) |*p| try p.applyRules(gpa, raw) else try gpa.dupe(u8, raw);
    defer gpa.free(text);
    var out_buf: [4096]u8 = undefined;
    var out = Io.File.Writer.init(.stdout(), io, &out_buf);
    try out.interface.print("{s}\n", .{text});
    try out.interface.flush();
}

fn msSince(io: Io, t0: Io.Timestamp) f64 {
    const d = t0.durationTo(Io.Clock.awake.now(io));
    return @as(f64, @floatFromInt(d.nanoseconds)) / 1e6;
}

/// `stt bench`: timings for loading, GPU setup, and repeated transcription.
fn cmdBench(gpa: std.mem.Allocator, io: Io, args: Args) !void {
    const samples = loadAudio(gpa, io, args);
    defer gpa.free(samples);
    var t0 = Io.Clock.awake.now(io);
    var model = loadModel(gpa, io, args.model);
    defer model.deinit();
    std.debug.print("model load: {d:.1} ms ({s})\n", .{ msSince(io, t0), args.model });
    const audio_s = @as(f64, @floatFromInt(samples.len)) / 16000.0;
    std.debug.print("audio: {d:.2} s\n", .{audio_s});

    t0 = Io.Clock.awake.now(io);
    var tr = openTranscriber(gpa, &model);
    defer tr.deinit();
    const g = tr.gpuTiming();
    std.debug.print("[gpu] {s}\n  setup (driver + PTX JIT + upload): {d:.1} ms; weights {d} MB as f16, upload DMA {d:.1} ms\n", .{ tr.deviceName(), msSince(io, t0), g.weight_mb, g.upload_ms });

    var best: f64 = std.math.inf(f64);
    var total: f64 = 0;
    var text: ?[]u8 = null;
    defer if (text) |x| gpa.free(x);
    for (0..args.runs) |r| {
        t0 = Io.Clock.awake.now(io);
        const t = try tr.transcribe(gpa, samples);
        const ms = msSince(io, t0);
        best = @min(best, ms);
        total += ms;
        if (r == 0) text = t else gpa.free(t);
        std.debug.print("  run {d}: {d:8.1} ms\n", .{ r + 1, ms });
    }
    std.debug.print("  best {d:.1} ms, mean {d:.1} ms = {d:.0}x realtime\n", .{ best, total / @as(f64, @floatFromInt(args.runs)), audio_s * 1000 / best });
    const st = tr.gpuTiming().stages;
    std.debug.print("  GPU time, last window: encoder {d:.2} ms, cross-KV {d:.2} ms, decode {d:.2} ms\n", .{ st.encode_ms, st.cross_ms, st.decode_ms });
    std.debug.print("  text: {s}\n", .{text.?});
}

fn usageExit() noreturn {
    std.debug.print("{s}", .{usage});
    std.process.exit(2);
}

fn needProfile(args: Args) []const u8 {
    return args.profile orelse {
        std.debug.print("error: this command needs a profile (drop --no-profile)\n", .{});
        std.process.exit(2);
    };
}

/// `stt train`: one training run on the GPU, right now (ignores the schedule).
fn cmdTrain(gpa: std.mem.Allocator, io: Io, args: Args) !void {
    const dir = needProfile(args);
    var model = loadModel(gpa, io, args.model);
    defer model.deinit();
    var p = try stt.Profile.load(gpa, io, dir);
    defer p.deinit();
    var job = try stt.training.collect(gpa, &p);
    defer job.deinit();
    std.debug.print("training {s}: {d} sample(s), {t}, {d} epochs, on the GPU...\n", .{ dir, job.samples.len, args.optimizer, args.epochs });
    const Report = struct {
        fn onStep(_: *anyopaque, s: stt.training.Step) void {
            if (s.epoch_done) std.debug.print("  epoch {d}/{d}  loss {d:.4}\n", .{ s.epoch, s.epochs, s.loss });
        }
    };
    var dummy: u8 = 0;
    const opts: stt.training.Options = .{ .epochs = args.epochs, .lr = args.lr, .optimizer = args.optimizer };
    const t0 = Io.Clock.awake.now(io);
    const r = stt.training.run(gpa, io, &model, &job, opts, .{ .ctx = &dummy, .onStep = Report.onStep }, null) catch |err| switch (err) {
        error.NoSamples => {
            std.debug.print("no samples yet: they come from feedback sent through Portal (see README).\n", .{});
            std.process.exit(1);
        },
        else => return err,
    };
    try stt.training.commit(&p, &job);
    std.debug.print("adapter v{d}: {d} sample(s), final loss {d:.4}, {d:.1} s\n", .{ r.version, r.samples, r.loss, msSince(io, t0) / 1000 });
}

fn cmdStatus(gpa: std.mem.Allocator, io: Io, args: Args) !void {
    const dir = needProfile(args);
    var p = try stt.Profile.load(gpa, io, dir);
    defer p.deinit();
    const cfg: stt.schedule.Config = .{};
    const w = cfg.window.?;
    std.debug.print(
        \\profile:   {s}
        \\learning:  {s}
        \\adapter:   v{d}
        \\samples:   {d} total, {d} new since the last training (training starts at {d})
        \\schedule:  daily {d:0>2}:{d:0>2}-{d:0>2}:{d:0>2} local time (Portal's Home can change it)
        \\
    , .{
        dir,                              if (p.learning) "on" else "off",
        p.version,                        p.next_sample - 1,
        stt.training.pending(&p),         cfg.min_new_samples,
        w.start_minute / 60,              w.start_minute % 60,
        w.end_minute / 60,                w.end_minute % 60,
    });
}

fn defaultProfileDir(arena: std.mem.Allocator, env: *const std.process.Environ.Map) ![]const u8 {
    if (env.get("LOCALAPPDATA")) |base| return std.fs.path.join(arena, &.{ base, "stt", "profile" });
    if (env.get("XDG_DATA_HOME")) |base| return std.fs.path.join(arena, &.{ base, "stt", "profile" });
    if (env.get("HOME")) |home| return std.fs.path.join(arena, &.{ home, ".local", "share", "stt", "profile" });
    return "profile";
}

