//! SttCoreML.app: stt's Core ML engine on an iPhone. Transcribes the
//! bundled testdata/hello.wav with the bundled model (default tiny.en) on
//! the Neural Engine, then on the GPU and CPU compute units for
//! comparison, and tries an adapter swap. Shows the log in a text view,
//! NSLogs it, and writes Documents/stt-coreml.txt. The first run generates
//! and compiles the programs into Library/Caches (seconds); later runs load
//! them from there.
const std = @import("std");
const stt = @import("stt");
const coreml = stt.coreml.lib;
const objc = coreml.objc;
const ce = stt.coreml.engine;
const Object = objc.Object;
const sel = objc.sel;
const msgSend = objc.msgSend;

extern fn UIApplicationMain(argc: c_int, argv: ?[*]?[*:0]u8, principal: ?*Object, delegate: ?*Object) c_int;
extern "c" fn _NSGetArgc() *c_int;
extern "c" fn _NSGetArgv() *?[*]?[*:0]u8;
extern fn NSSearchPathForDirectoriesInDomains(directory: usize, domains: usize, expand: objc.BOOL) ?*Object;

const CGRect = extern struct { x: f64, y: f64, w: f64, h: f64 };

const G = struct {
    var gpa: std.mem.Allocator = undefined;
    var io: std.Io = undefined;
    var text_view: ?*Object = null;
    var window: ?*Object = null;
    var lock: std.Io.Mutex = .init;
    var text: std.ArrayList(u8) = .empty;
};

pub fn main(init: std.process.Init) void {
    G.gpa = init.gpa;
    G.io = init.io;
    const pool = objc.AutoreleasePool.push();
    const cb = objc.ClassBuilder.begin(objc.getClass("UIResponder"), "SttCoreMLAppDelegate") orelse @panic("delegate class exists");
    cb.addMethod("application:didFinishLaunchingWithOptions:", &didFinishLaunching, "c@:@@");
    _ = cb.register();
    const name = objc.newString("SttCoreMLAppDelegate");
    pool.pop();
    _ = UIApplicationMain(_NSGetArgc().*, _NSGetArgv().*, null, name);
}

fn didFinishLaunching(_: *Object, _: *objc.Selector, _: ?*Object, _: ?*Object) callconv(.c) objc.BOOL {
    const screen = msgSend(?*Object, objc.getClass("UIScreen"), sel("mainScreen"), .{});
    const bounds = msgSend(CGRect, screen, sel("bounds"), .{});
    const win = msgSend(?*Object, msgSend(?*Object, objc.getClass("UIWindow"), sel("alloc"), .{}), sel("initWithFrame:"), .{bounds});
    const vc = msgSend(?*Object, msgSend(?*Object, objc.getClass("UIViewController"), sel("alloc"), .{}), sel("init"), .{});
    const tv = msgSend(?*Object, msgSend(?*Object, objc.getClass("UITextView"), sel("alloc"), .{}), sel("initWithFrame:"), .{bounds});
    msgSend(void, tv, sel("setEditable:"), .{@as(objc.BOOL, 0)});
    msgSend(void, vc, sel("setView:"), .{tv});
    msgSend(void, win, sel("setRootViewController:"), .{vc});
    msgSend(void, win, sel("makeKeyAndVisible"), .{});
    G.window = win;
    G.text_view = tv;
    say("stt Core ML check: running (also written to Documents/stt-coreml.txt)", .{});
    objc.onBackgroundAsync(null, &work);
    return 1;
}

fn say(comptime fmt: []const u8, args: anytype) void {
    var buf: [1024]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, fmt, args) catch buf[0..];
    objc.log(s);
    G.lock.lockUncancelable(G.io);
    G.text.appendSlice(G.gpa, s) catch {};
    G.text.append(G.gpa, '\n') catch {};
    G.lock.unlock(G.io);
    objc.onMainAsync(null, &showText);
}

fn showText(_: ?*anyopaque) callconv(.c) void {
    const pool = objc.AutoreleasePool.push();
    defer pool.pop();
    G.lock.lockUncancelable(G.io);
    const s = objc.newString(G.text.items);
    G.lock.unlock(G.io);
    defer if (s) |x| objc.release(x);
    msgSend(void, G.text_view, sel("setText:"), .{s});
}

fn cachesDir() []const u8 {
    // NSCachesDirectory = 13, NSUserDomainMask = 1
    const list = NSSearchPathForDirectoriesInDomains(13, 1, 1);
    if (objc.arrayCount(list) == 0) return objc.tempDir();
    return objc.stringBytes(objc.arrayAt(list, 0));
}

fn work(_: ?*anyopaque) callconv(.c) void {
    const pool = objc.AutoreleasePool.push();
    defer pool.pop();
    run() catch |e| say("FAILED: {t} ({s})", .{ e, coreml.runtime.lastError() });
    const gpa = G.gpa;
    const out = std.fmt.allocPrint(gpa, "{s}/stt-coreml.txt", .{objc.documentsDir()}) catch return;
    defer gpa.free(out);
    G.lock.lockUncancelable(G.io);
    defer G.lock.unlock(G.io);
    std.Io.Dir.cwd().writeFile(G.io, .{ .sub_path = out, .data = G.text.items }) catch {};
}

fn msSince(t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(G.io).nanoseconds - t0.nanoseconds)) / 1e6;
}

fn run() !void {
    const gpa = G.gpa;
    const io = G.io;
    const res = objc.bundleResourcePath();
    const model_path = try std.fmt.allocPrint(gpa, "{s}/stt/model.bin", .{res});
    defer gpa.free(model_path);
    const wav_path = try std.fmt.allocPrint(gpa, "{s}/stt/hello.wav", .{res});
    defer gpa.free(wav_path);
    const cache = try std.fmt.allocPrint(gpa, "{s}/stt-coreml", .{cachesDir()});
    defer gpa.free(cache);

    if (coreml.runtime.availableDevices()) |d| say("devices: cpu={} gpu={} neural_engine={} (ANE cores {d})", .{ d.cpu, d.gpu, d.neural_engine, d.ane_cores });
    var t0 = std.Io.Clock.awake.now(io);
    var model = try stt.Model.load(gpa, io, model_path);
    defer model.deinit();
    const samples = try stt.audio.load(gpa, io, wav_path, .{});
    defer gpa.free(samples);
    say("model: state {d}, layers {d}/{d}, loaded in {d:.0} ms; audio {d:.2} s", .{ model.hp.n_text_state, model.hp.n_audio_layer, model.hp.n_text_layer, msSince(t0), @as(f64, @floatFromInt(samples.len)) / 16000.0 });

    var first_text: ?[]u8 = null;
    defer if (first_text) |t| gpa.free(t);
    for ([_]ce.ComputeUnits{ .cpu_and_neural_engine, .cpu_and_gpu, .cpu_only }) |units| {
        t0 = std.Io.Clock.awake.now(io);
        var tr = stt.Transcriber.init(gpa, &model, .{ .backend = .{ .coreml = .{ .cache_dir = cache, .compute_units = units } } }) catch |e| {
            say("[{t}] init failed: {t} ({s})", .{ units, e, coreml.runtime.lastError() });
            continue;
        };
        defer tr.deinit();
        const e = tr.engine.coreml;
        say("[{t}] ready in {d:.0} ms: encoder {d:.0} ms{s}, decoder {d:.0} ms{s}", .{
            units,                       msSince(t0),
            e.timing.encoder_prepare_ms, if (e.timing.encoder_from_cache) " (cache)" else " (generated + compiled)",
            e.timing.decoder_prepare_ms, if (e.timing.decoder_from_cache) " (cache)" else " (generated + compiled)",
        });
        for (0..3) |r| {
            t0 = std.Io.Clock.awake.now(io);
            const text = try tr.transcribe(gpa, samples);
            say("[{t}] run {d}: {d:.0} ms (log-mel {d:.1}, encoder {d:.1} ms, {d} steps {d:.2} ms/step): \"{s}\"", .{
                units,                     r + 1,              msSince(t0), e.timing.mel_ms, e.timing.encode_ms, e.timing.steps,
                e.timing.steps_ms / @as(f64, @floatFromInt(@max(e.timing.steps, 1))), text,
            });
            if (first_text == null) first_text = text else {
                if (!std.mem.eql(u8, text, first_text.?)) say("  (differs from the Neural Engine transcript)", .{});
                gpa.free(text);
            }
        }
        if (units == .cpu_and_neural_engine) {
            var ad = try stt.Adapter.init(gpa, &model, 8, 16, 1);
            defer ad.deinit(gpa);
            for (ad.loras) |lo| for (lo.b, 0..) |*x, j| {
                x.* = 0.002 * @sin(@as(f32, @floatFromInt(j)) * 0.37);
            };
            t0 = std.Io.Clock.awake.now(io);
            try tr.setAdapter(&ad, 7);
            say("[ane] setAdapter (decoder regenerated + compiled): {d:.0} ms{s}", .{ msSince(t0), if (e.timing.decoder_from_cache) " (cache)" else "" });
            const t2 = try tr.transcribe(gpa, samples);
            defer gpa.free(t2);
            say("  with adapter: \"{s}\"", .{t2});
            try tr.setAdapter(null, 0);
        }
    }
    say("done", .{});
}
