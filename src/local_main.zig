//! `stt-local`: the laptop program Portal runs as a child process. It wraps
//! `stt.Local` (transcription on the laptop's NPU) in a JSON-lines protocol
//! on stdin/stdout. The protocol (proto 1) is agreed with Portal:
//!
//! Framing
//!   - stdout carries only protocol lines: UTF-8, one JSON object per line,
//!     at most 8 MiB per line. Logs go to stderr.
//!   - The first stdout line is
//!       {"event":"ready","proto":1,"version":"0.1.0","backend":"htp|cpu|cuda","model":"<file name>","adapter_version":N}
//!   - Every request carries "rid" (u32) and its reply echoes it. Replies may
//!     in principle come out of order; this version answers in order.
//!     Events carry no rid.
//!   - Errors: {"rid":N,"error":"<code>","message":"..."} with code one of
//!     bad_request, bad_audio, too_long, busy, backend_lost, unknown_id,
//!     not_found, internal. Malformed JSON gets rid 0. backend_lost means the
//!     NPU/GPU engine failed for good: Portal should restart the process.
//!
//! Requests -> replies
//!   {"rid":1,"op":"transcribe","pcm_s16le_b64":"..."}     raw 16 kHz mono s16le (preferred)
//!   {"rid":1,"op":"transcribe","wav_b64":"..."}           or a WAV/MP3/QOA file
//!       -> {"rid":1,"id":7,"text":"..."}                   id 0 = nothing heard (no feedback possible)
//!       at most 120 s of audio, else error too_long
//!   {"rid":2,"op":"feedback","id":7,"text":"...","ok":true}
//!       -> {"rid":2,"outcome":"learned","sample_id":3,"new_rules":1,"new_words":2}
//!        | {"rid":2,"outcome":"discarded"}                 (unknown/expired id: error unknown_id)
//!   {"rid":3,"op":"export_bundle","max_bytes":8388608}
//!       -> {"rid":3,"b64":"...","through":12,"count":4,"version":2}  | {"rid":3,"b64":null,"count":0,...}
//!   {"rid":4,"op":"commit_export","through":12}            -> {"rid":4,"ok":true}
//!   {"rid":5,"op":"import_update","b64":"..."}             -> {"rid":5,"version":3}
//!   {"rid":6,"op":"set_learning","on":false}               -> {"rid":6,"ok":true}
//!   {"rid":7,"op":"status"}                                -> {"rid":7,"proto":1,...}
//!
//! Events
//!   {"event":"wakeup","deadline_unix_ms":N|null}   whenever the tick deadline
//!   changes. The program calls tick itself at the deadline (a timer thread
//!   that sleeps until exactly then; no fixed-interval polling). The only
//!   timed work is auto-confirming transcripts after 60 s of silence.
//!
//! Lifetime: Portal closes stdin to stop the program; it then confirms
//! pending transcripts (silence = correct), frees the engine and exits 0.
//!
//! Sandbox: no network, sockets or child processes; reads the model and
//! adapter packages; writes only under --profile.
//!
//! Usage: stt-local --model <path> --profile <dir> [--adapter <path>]
//!          [--adapter-version <n>] [--qnn <dir>] [--backend htp|cpu|cuda]
//!          [--device-name <name>]
const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const stt = @import("stt");

pub const version = "0.1.0";
pub const proto = 1;
/// Longest protocol line, in or out.
pub const max_line = 8 << 20;
/// Longest audio accepted per request.
pub const max_seconds = 120;
/// Bundles are capped so their base64 line stays under `max_line`.
const max_bundle_bytes = 5 << 20;

pub const ErrorCode = enum { bad_request, bad_audio, too_long, busy, backend_lost, unknown_id, not_found, internal };

const usage =
    \\stt-local - local speech to text on the laptop NPU, for Portal (JSON lines on stdin/stdout)
    \\
    \\USAGE
    \\  stt-local --model <path> --profile <dir> [options]
    \\
    \\OPTIONS
    \\  --model <path>          Model file (the stt-model data package)
    \\  --profile <dir>         Writable data folder (the only place written to)
    \\  --adapter <path>        Adapter file or update (the stt-adapter data package)
    \\  --adapter-version <n>   Version of a raw --adapter file (default 0)
    \\  --qnn <dir>             Folder with the QNN backend DLLs (default: search)
    \\  --backend <b>           htp (default), cpu (QNN CPU backend) or cuda (NVIDIA GPU, for tests)
    \\  --device-name <name>    Name on this device's sync bundles (default laptop)
    \\
;

const BackendArg = enum { htp, cpu, cuda };

const Args = struct {
    model: ?[]const u8 = null,
    profile: ?[]const u8 = null,
    adapter: ?[]const u8 = null,
    adapter_version: u32 = 0,
    qnn: ?[]const u8 = null,
    backend: BackendArg = .htp,
    device_name: []const u8 = "laptop",
};

fn parseArgs(argv: []const [:0]const u8) !Args {
    var a: Args = .{};
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg: []const u8 = argv[i];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) return error.Usage;
        if (i + 1 >= argv.len) return error.Usage;
        const val: []const u8 = argv[i + 1];
        i += 1;
        if (std.mem.eql(u8, arg, "--model")) {
            a.model = val;
        } else if (std.mem.eql(u8, arg, "--profile")) {
            a.profile = val;
        } else if (std.mem.eql(u8, arg, "--adapter")) {
            a.adapter = val;
        } else if (std.mem.eql(u8, arg, "--adapter-version")) {
            a.adapter_version = std.fmt.parseInt(u32, val, 10) catch return error.Usage;
        } else if (std.mem.eql(u8, arg, "--qnn")) {
            a.qnn = val;
        } else if (std.mem.eql(u8, arg, "--backend")) {
            a.backend = std.meta.stringToEnum(BackendArg, val) orelse return error.Usage;
        } else if (std.mem.eql(u8, arg, "--device-name")) {
            a.device_name = val;
        } else return error.Usage;
    }
    if (a.model == null or a.profile == null) return error.Usage;
    return a;
}

/// Serialized writes of whole lines to stdout.
const Out = struct {
    io: Io,
    lock: Io.Mutex = .init,
    w: *Io.Writer,

    fn line(o: *Out, v: anytype) void {
        o.lock.lockUncancelable(o.io);
        defer o.lock.unlock(o.io);
        std.json.Stringify.value(v, .{}, o.w) catch return;
        o.w.writeByte('\n') catch return;
        o.w.flush() catch return;
    }

    fn fail(o: *Out, rid: u32, code: ErrorCode, message: []const u8) void {
        o.line(.{ .rid = rid, .@"error" = @tagName(code), .message = message });
    }
};

/// Calls `tick` at the deadline and reports deadline changes as events.
/// It only sleeps and dispatches; no compute happens on it.
const Timer = struct {
    io: Io,
    out: *Out,
    local: *stt.Local,
    mu: Io.Mutex = .init,
    cond: Io.Condition = .init,
    deadline: ?i64 = null,
    reported: ?i64 = null,
    quit: bool = false,

    fn onWakeup(ctx: *anyopaque, d: ?i64) void {
        const t: *Timer = @ptrCast(@alignCast(ctx));
        t.mu.lockUncancelable(t.io);
        defer t.mu.unlock(t.io);
        t.setLocked(d);
    }

    fn setLocked(t: *Timer, d: ?i64) void {
        t.deadline = d;
        t.cond.signal(t.io);
        if (std.meta.eql(d, t.reported)) return;
        t.reported = d;
        t.out.line(.{ .event = "wakeup", .deadline_unix_ms = d });
    }

    fn run(t: *Timer) void {
        t.mu.lockUncancelable(t.io);
        defer t.mu.unlock(t.io);
        while (!t.quit) {
            const d = t.deadline orelse {
                t.cond.wait(t.io, &t.mu) catch {};
                continue;
            };
            const left = d - nowMs(t.io);
            if (left > 0) {
                t.cond.waitTimeout(t.io, &t.mu, .{ .duration = .{ .raw = .fromMilliseconds(left), .clock = .awake } }) catch {};
                continue;
            }
            t.deadline = null;
            t.mu.unlock(t.io);
            const next = t.local.tick();
            t.mu.lockUncancelable(t.io);
            t.setLocked(next);
        }
    }

    fn stop(t: *Timer) void {
        t.mu.lockUncancelable(t.io);
        defer t.mu.unlock(t.io);
        t.quit = true;
        t.cond.signal(t.io);
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    const args = parseArgs(argv) catch {
        std.debug.print("{s}", .{usage});
        std.process.exit(2);
    };

    var out_buf: [64 << 10]u8 = undefined;
    var stdout_w = Io.File.Writer.init(.stdout(), io, &out_buf);
    var out: Out = .{ .io = io, .w = &stdout_w.interface };

    const local = stt.Local.open(gpa, io, .{
        .model_path = args.model.?,
        .profile_dir = args.profile.?,
        .adapter_path = args.adapter,
        .adapter_version = args.adapter_version,
        .backend = if (args.backend == .cuda) .cuda else .npu,
        .npu = .{ .qnn_dir = args.qnn, .backend = if (args.backend == .cpu) .cpu else .htp },
        .device_name = args.device_name,
        .service = .{ .max_seconds = max_seconds },
    }) catch |err| {
        std.log.err("stt-local: cannot start ({t})", .{err});
        std.process.exit(1);
    };
    const model_id = std.fs.path.basename(args.model.?);

    out.line(.{
        .event = "ready",
        .proto = proto,
        .version = version,
        .backend = @tagName(args.backend),
        .model = model_id,
        .adapter_version = local.adapterVersion(),
    });

    var timer: Timer = .{ .io = io, .out = &out, .local = local };
    const thread = std.Thread.spawn(.{}, Timer.run, .{&timer}) catch |err| {
        std.log.err("stt-local: cannot start the timer ({t})", .{err});
        local.close();
        std.process.exit(1);
    };
    local.setWakeup(.{ .ctx = &timer, .onWakeup = Timer.onWakeup });

    const in_buf = try gpa.alloc(u8, max_line + 1);
    defer gpa.free(in_buf);
    var stdin_r = Io.File.Reader.initStreaming(.stdin(), io, in_buf);
    const r = &stdin_r.interface;
    const ctx: Ctx = .{ .gpa = gpa, .local = local, .out = &out, .backend = args.backend, .model_id = model_id };
    while (true) {
        const line = (r.takeDelimiter('\n') catch |err| switch (err) {
            error.StreamTooLong => {
                out.fail(0, .bad_request, "line longer than 8 MiB");
                _ = r.discardDelimiterInclusive('\n') catch break;
                continue;
            },
            error.ReadFailed => break,
        }) orelse break; // stdin closed: shut down
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;
        ctx.handle(trimmed);
    }

    local.setWakeup(null);
    timer.stop();
    thread.join();
    local.close(); // confirms pending transcripts
}

const Ctx = struct {
    gpa: Allocator,
    local: *stt.Local,
    out: *Out,
    backend: BackendArg,
    model_id: []const u8,

    fn handle(c: Ctx, line: []const u8) void {
        var arena_state: std.heap.ArenaAllocator = .init(c.gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const v = std.json.parseFromSliceLeaky(std.json.Value, arena, line, .{}) catch
            return c.out.fail(0, .bad_request, "malformed JSON");
        if (v != .object) return c.out.fail(0, .bad_request, "request must be a JSON object");
        const req: Req = .{ .obj = v.object };
        const rid = req.int(u32, "rid") orelse return c.out.fail(0, .bad_request, "missing or invalid rid");
        var msg: []const u8 = "";
        c.dispatch(arena, req, rid, &msg) catch |err| {
            const code = codeOf(err);
            c.out.fail(rid, code, if (msg.len > 0) msg else @errorName(err));
        };
    }

    fn dispatch(c: Ctx, arena: Allocator, req: Req, rid: u32, msg: *[]const u8) !void {
        const op = req.str("op") orelse {
            msg.* = "missing op";
            return error.BadRequest;
        };
        if (std.mem.eql(u8, op, "transcribe")) return c.opTranscribe(arena, req, rid, msg);
        if (std.mem.eql(u8, op, "feedback")) return c.opFeedback(req, rid, msg);
        if (std.mem.eql(u8, op, "export_bundle")) {
            const want = req.int(usize, "max_bytes") orelse stt.sync.default_bundle_bytes;
            const b = (try c.local.exportBundle(arena, @min(want, max_bundle_bytes))) orelse
                return c.out.line(.{ .rid = rid, .b64 = @as(?[]const u8, null), .through = @as(u32, 0), .count = @as(u32, 0), .version = c.local.status().profile_version });
            const enc = std.base64.standard.Encoder;
            const b64 = try arena.alloc(u8, enc.calcSize(b.bytes.len));
            _ = enc.encode(b64, b.bytes);
            return c.out.line(.{ .rid = rid, .b64 = @as(?[]const u8, b64), .through = b.through, .count = b.count, .version = c.local.status().profile_version });
        }
        if (std.mem.eql(u8, op, "commit_export")) {
            const through = req.int(u32, "through") orelse {
                msg.* = "missing through";
                return error.BadRequest;
            };
            try c.local.commitExport(through);
            return c.out.line(.{ .rid = rid, .ok = true });
        }
        if (std.mem.eql(u8, op, "import_update")) {
            const bytes = try decodeB64(arena, req.str("b64") orelse {
                msg.* = "missing b64";
                return error.BadRequest;
            }, msg);
            const ver = c.local.importUpdate(arena, bytes) catch |err| switch (err) {
                error.BadUpdate, error.Truncated, error.BadAdapterFile, error.AdapterModelMismatch => {
                    msg.* = "not a valid update for this model";
                    return error.BadRequest;
                },
                else => return err,
            };
            return c.out.line(.{ .rid = rid, .version = ver, .adapter_version = c.local.adapterVersion() });
        }
        if (std.mem.eql(u8, op, "set_learning")) {
            const on = req.boolean("on") orelse {
                msg.* = "missing on";
                return error.BadRequest;
            };
            try c.local.setLearning(on);
            return c.out.line(.{ .rid = rid, .ok = true });
        }
        if (std.mem.eql(u8, op, "status")) {
            const s = c.local.status();
            return c.out.line(.{
                .rid = rid,
                .proto = proto,
                .version = version,
                .backend = @tagName(c.backend),
                .device = s.device,
                .model = c.model_id,
                .adapter_version = s.adapter_version,
                .profile_version = s.profile_version,
                .learning = s.learning,
                .pending_export = s.pending_export,
                .deadline_unix_ms = s.deadline_unix_ms,
            });
        }
        msg.* = "unknown op";
        return error.BadRequest;
    }

    fn opTranscribe(c: Ctx, arena: Allocator, req: Req, rid: u32, msg: *[]const u8) !void {
        const res = if (req.str("pcm_s16le_b64")) |b64| blk: {
            const n_bytes = std.base64.standard.Decoder.calcSizeForSlice(b64) catch {
                msg.* = "invalid base64";
                return error.BadRequest;
            };
            if (n_bytes / 2 > max_seconds * stt.audio.target_rate) return error.AudioTooLong;
            const bytes = try decodeB64(arena, b64, msg);
            if (bytes.len % 2 != 0) {
                msg.* = "odd number of bytes in s16le PCM";
                return error.BadAudio;
            }
            const samples = try arena.alloc(f32, bytes.len / 2);
            for (samples, 0..) |*s, i| {
                const x = std.mem.readInt(i16, bytes[i * 2 ..][0..2], .little);
                s.* = @as(f32, @floatFromInt(x)) / 32768.0;
            }
            break :blk try c.local.transcribePcm(arena, samples);
        } else if (req.str("wav_b64")) |b64| blk: {
            const bytes = try decodeB64(arena, b64, msg);
            break :blk try c.local.transcribe(arena, bytes);
        } else {
            msg.* = "transcribe needs pcm_s16le_b64 or wav_b64";
            return error.BadRequest;
        };
        c.out.line(.{ .rid = rid, .id = res.id, .text = try validUtf8(arena, res.text) });
    }

    fn opFeedback(c: Ctx, req: Req, rid: u32, msg: *[]const u8) !void {
        const id = req.int(u64, "id") orelse {
            msg.* = "missing id";
            return error.BadRequest;
        };
        const ok = req.boolean("ok") orelse {
            msg.* = "missing ok";
            return error.BadRequest;
        };
        const text = req.str("text") orelse "";
        switch (try c.local.feedback(id, text, ok)) {
            .learned => |l| c.out.line(.{ .rid = rid, .outcome = "learned", .sample_id = l.sample_id, .new_rules = l.new_rules, .new_words = l.new_words }),
            .discarded => c.out.line(.{ .rid = rid, .outcome = "discarded" }),
            .unknown_id => return error.UnknownId,
            .no_profile => return error.NoProfile,
            .failed => |e| return e,
        }
    }
};

/// Field access on a request object, tolerant of wrong types (null).
const Req = struct {
    obj: std.json.ObjectMap,

    fn str(r: Req, key: []const u8) ?[]const u8 {
        const v = r.obj.get(key) orelse return null;
        return if (v == .string) v.string else null;
    }

    fn int(r: Req, comptime T: type, key: []const u8) ?T {
        const v = r.obj.get(key) orelse return null;
        if (v != .integer) return null;
        return std.math.cast(T, v.integer);
    }

    fn boolean(r: Req, key: []const u8) ?bool {
        const v = r.obj.get(key) orelse return null;
        return if (v == .bool) v.bool else null;
    }
};

fn decodeB64(arena: Allocator, b64: []const u8, msg: *[]const u8) ![]u8 {
    const dec = std.base64.standard.Decoder;
    const n = dec.calcSizeForSlice(b64) catch {
        msg.* = "invalid base64";
        return error.BadRequest;
    };
    const buf = try arena.alloc(u8, n);
    dec.decode(buf, b64) catch {
        msg.* = "invalid base64";
        return error.BadRequest;
    };
    return buf;
}

fn codeOf(err: anyerror) ErrorCode {
    return switch (err) {
        error.BadRequest => .bad_request,
        error.BadAudio => .bad_audio,
        error.AudioTooLong => .too_long,
        error.BackendLost => .backend_lost,
        error.UnknownId => .unknown_id,
        error.FileNotFound => .not_found,
        else => .internal,
    };
}

/// Model output can end mid-character; JSON needs valid UTF-8.
fn validUtf8(arena: Allocator, s: []const u8) ![]const u8 {
    if (std.unicode.utf8ValidateSlice(s)) return s;
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < s.len) {
        const n = std.unicode.utf8ByteSequenceLength(s[i]) catch 0;
        if (n > 0 and i + n <= s.len and std.unicode.utf8ValidateSlice(s[i..][0..n])) {
            try out.appendSlice(arena, s[i..][0..n]);
            i += n;
        } else {
            try out.appendSlice(arena, "\u{FFFD}");
            i += 1;
        }
    }
    return out.items;
}

fn nowMs(io: Io) i64 {
    return @intCast(@divFloor(Io.Clock.real.now(io).nanoseconds, std.time.ns_per_ms));
}

test "validUtf8 replaces broken sequences" {
    var a: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer a.deinit();
    try std.testing.expectEqualStrings("ok", try validUtf8(a.allocator(), "ok"));
    try std.testing.expectEqualStrings("a\u{FFFD}b", try validUtf8(a.allocator(), "a\xe2b"));
}
