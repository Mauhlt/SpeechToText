//! The home desktop: everything Portal talks to, in one object.
//!
//! Portal passes audio in and gets text out, relays the three feedback
//! responses, and moves sync bundles/updates. Training runs on the GPU on a
//! background thread, started by the schedule (or `trainNow`);
//! `trainingStatus` and the listener tell Portal when it is running so it can
//! warn the user that GPU work will be slower.
//!
//! Event-driven, no polling: `tick` returns when it next needs to run (a
//! silent transcript to auto-confirm, or the training window opening), and
//! the optional wakeup hook reports whenever that deadline changes. A
//! training run finishes on its own thread (no tick needed for that).
//!
//! All methods are thread-safe (serialized by one lock); the training run
//! itself happens outside the lock, so transcription keeps working (more
//! slowly) while it trains.
const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const Model = @import("model.zig").Model;
const Profile = @import("profile.zig").Profile;
const transcriber = @import("transcriber.zig");
const Transcriber = transcriber.Transcriber;
const service = @import("service.zig");
const Service = service.Service;
const sync = @import("sync.zig");
const training = @import("training.zig");
const schedule = @import("schedule.zig");
const Adapter = @import("adapter.zig").Adapter;

pub const Config = struct {
    model_path: []const u8,
    profile_dir: []const u8,
    service: service.Options = .{},
    transcriber: transcriber.Options = .{},
    schedule: schedule.Config = .{},
    training: training.Options = .{},
    storage: sync.StoragePolicy = .{},
    /// Vocabulary hint added in front of the learned words.
    base_prompt: []const u8 = "",
};

pub const Result = service.Result;
pub const Outcome = service.Outcome;

/// Told when `tick` should next run: `deadline_unix_ms` (null = nothing
/// pending; wait for the next call into Home). Called after any change that
/// can move the deadline, possibly from the training thread; keep it quick
/// and don't call back into `Home` from it (schedule a wakeup instead).
pub const Wakeup = struct {
    ctx: *anyopaque,
    onWakeup: *const fn (ctx: *anyopaque, deadline_unix_ms: ?i64) void,
};

pub const Home = struct {
    gpa: Allocator,
    io: Io,
    cfg: Config,
    lock: Io.Mutex = .init,
    model: Model,
    profile: Profile,
    tr: Transcriber,
    svc: Service,
    listener: ?schedule.Listener = null,
    wakeup: ?Wakeup = null,

    status_lock: Io.Mutex = .init,
    status: schedule.Status = .{},
    // Training run. `running` is guarded by `lock`; the thread handle is kept
    // until the next run (or close) joins it.
    worker: ?std.Thread = null,
    running: bool = false,
    job: ?training.Job = null,
    cancel_flag: std.atomic.Value(bool) = .init(false),

    pub fn open(gpa: Allocator, io: Io, cfg: Config) !*Home {
        const h = try gpa.create(Home);
        errdefer gpa.destroy(h);
        h.* = .{
            .gpa = gpa,
            .io = io,
            .cfg = cfg,
            .model = undefined,
            .profile = undefined,
            .tr = undefined,
            .svc = undefined,
        };
        h.model = try Model.load(gpa, io, cfg.model_path);
        errdefer h.model.deinit();
        h.profile = try Profile.load(gpa, io, cfg.profile_dir);
        errdefer h.profile.deinit();
        h.tr = try Transcriber.init(gpa, &h.model, cfg.transcriber);
        errdefer h.tr.deinit();
        if (try training.loadAdapter(gpa, &h.profile, &h.model)) |ad_| {
            var ad = ad_;
            defer ad.deinit(gpa);
            try h.tr.setAdapter(&ad, h.profile.version);
        }
        const pt = try h.profile.promptText(gpa, cfg.base_prompt);
        defer gpa.free(pt);
        try h.tr.setPrompt(pt);
        h.svc = try Service.init(gpa, io, &h.tr, &h.profile, cfg.service);
        h.lock.lockUncancelable(io);
        h.refreshStatusLocked();
        h.lock.unlock(io);
        return h;
    }

    /// Stops any training run (the previous adapter stays), confirms pending
    /// transcripts (silence = correct) and frees everything.
    pub fn close(h: *Home) void {
        h.cancel_flag.store(true, .release);
        if (h.worker) |w| w.join();
        if (h.job) |*j| j.deinit();
        h.svc.deinit();
        h.tr.deinit();
        h.profile.deinit();
        h.model.deinit();
        h.gpa.destroy(h);
    }

    pub fn setListener(h: *Home, l: ?schedule.Listener) void {
        h.lock.lockUncancelable(h.io);
        defer h.lock.unlock(h.io);
        h.listener = l;
    }

    /// Sets the wakeup hook and reports the current deadline right away.
    pub fn setWakeup(h: *Home, w: ?Wakeup) void {
        h.lock.lockUncancelable(h.io);
        h.wakeup = w;
        h.lock.unlock(h.io);
        h.notify();
    }

    pub fn setSchedule(h: *Home, cfg: schedule.Config) void {
        {
            h.lock.lockUncancelable(h.io);
            defer h.lock.unlock(h.io);
            h.cfg.schedule = cfg;
            h.refreshStatusLocked();
        }
        h.notify();
    }

    // ------------------------------------------------------------ transcription

    /// Transcribes an audio file's bytes (WAV, MP3 or QOA; any rate/channels)
    /// sent by `device`. Caller owns `text`.
    pub fn transcribe(h: *Home, gpa: Allocator, audio_bytes: []const u8, device: []const u8) !Result {
        const r = blk: {
            h.lock.lockUncancelable(h.io);
            defer h.lock.unlock(h.io);
            break :blk try h.svc.transcribeAudio(gpa, audio_bytes, device);
        };
        h.notify(); // a new transcript is waiting for feedback / silence
        return r;
    }

    /// One of the three responses for transcript `id`: fix (corrected text,
    /// true), discard (_, false). Silence needs no call.
    pub fn feedback(h: *Home, id: u64, text: []const u8, success: bool) !Outcome {
        const out = blk: {
            h.lock.lockUncancelable(h.io);
            defer h.lock.unlock(h.io);
            const out = try h.svc.feedback(id, text, success);
            h.refreshStatusLocked();
            break :blk out;
        };
        h.notify();
        return out;
    }

    /// "nothing": stop keeping audio and learning (or resume).
    pub fn setLearning(h: *Home, on: bool) !void {
        {
            h.lock.lockUncancelable(h.io);
            defer h.lock.unlock(h.io);
            try h.profile.setLearning(on);
            h.refreshStatusLocked();
        }
        h.notify();
    }

    // ------------------------------------------------------------ sync

    pub fn importBundle(h: *Home, gpa: Allocator, bytes: []const u8) !sync.ImportReport {
        const r = blk: {
            h.lock.lockUncancelable(h.io);
            defer h.lock.unlock(h.io);
            const r = try sync.importBundle(gpa, &h.profile, bytes);
            h.refreshStatusLocked();
            break :blk r;
        };
        h.notify();
        return r;
    }

    pub fn exportUpdate(h: *Home, gpa: Allocator, device_version: u32) !?[]u8 {
        h.lock.lockUncancelable(h.io);
        defer h.lock.unlock(h.io);
        return sync.exportUpdate(gpa, &h.profile, device_version);
    }

    // ------------------------------------------------------------ training

    pub fn trainingStatus(h: *Home) schedule.Status {
        h.status_lock.lockUncancelable(h.io);
        defer h.status_lock.unlock(h.io);
        return h.status;
    }

    /// Starts training now, ignoring the time window (needs at least one sample).
    pub fn trainNow(h: *Home) !void {
        {
            h.lock.lockUncancelable(h.io);
            defer h.lock.unlock(h.io);
            if (h.running) return error.TrainingAlreadyRunning;
            try h.startLocked();
        }
        h.notify();
    }

    /// Asks a running training to stop; the previous adapter stays in use.
    pub fn cancelTraining(h: *Home) void {
        h.cancel_flag.store(true, .release);
    }

    /// Does whatever is due: confirms transcripts whose silence period has
    /// passed and starts scheduled training. Returns the unix-ms time it next
    /// needs to run, or null if nothing is pending (the wakeup hook reports
    /// later changes). Calling it early is harmless.
    pub fn tick(h: *Home) ?i64 {
        h.lock.lockUncancelable(h.io);
        defer h.lock.unlock(h.io);
        h.svc.tick();
        if (!h.running and h.shouldStartLocked()) {
            h.startLocked() catch |err| h.setError(err);
        }
        h.refreshStatusLocked();
        return h.deadlineLocked();
    }

    /// When `tick` next has work: the earliest of the next silence
    /// auto-confirm and the training window opening (if enough samples wait).
    fn deadlineLocked(h: *Home) ?i64 {
        const now_ms = nowMs(h.io);
        var best: ?i64 = null;
        if (h.svc.msUntilTick()) |ms| best = now_ms + ms;
        if (!h.running and h.cfg.schedule.window != null and h.profile.learning and
            training.pending(&h.profile) >= @max(1, h.cfg.schedule.min_new_samples))
        {
            const w = h.cfg.schedule.window.?;
            const wait_min: i64 = w.minutesUntilOpen(schedule.localMinute(h.io));
            const at = now_ms + wait_min * 60_000;
            best = if (best) |b| @min(b, at) else at;
        }
        return best;
    }

    fn notify(h: *Home) void {
        const w, const deadline = blk: {
            h.lock.lockUncancelable(h.io);
            defer h.lock.unlock(h.io);
            break :blk .{ h.wakeup orelse return, h.deadlineLocked() };
        };
        w.onWakeup(w.ctx, deadline);
    }

    fn shouldStartLocked(h: *Home) bool {
        const w = h.cfg.schedule.window orelse return false;
        if (!h.profile.learning) return false;
        if (training.pending(&h.profile) < @max(1, h.cfg.schedule.min_new_samples)) return false;
        return w.contains(schedule.localMinute(h.io));
    }

    fn startLocked(h: *Home) !void {
        if (h.profile.next_sample <= 1) return error.NoSamples;
        // The previous run's thread has finished its work (running == false);
        // reap it before starting a new one.
        if (h.worker) |w| {
            w.join();
            h.worker = null;
        }
        h.job = try training.collect(h.gpa, &h.profile);
        errdefer {
            h.job.?.deinit();
            h.job = null;
        }
        h.cancel_flag.store(false, .release);
        {
            h.status_lock.lockUncancelable(h.io);
            defer h.status_lock.unlock(h.io);
            h.status.state = .running;
            h.status.started_unix = schedule.nowUnix(h.io);
            h.status.progress = 0;
            h.status.eta_seconds = null;
            h.status.epoch = 0;
            h.status.epochs = @intCast(h.cfg.training.epochs);
            h.status.next_start_unix = null;
        }
        h.worker = try std.Thread.spawn(.{}, work, .{h});
        h.running = true;
        h.emit(.{ .started = h.trainingStatus() });
    }

    /// The training thread: GPU work outside the lock (reads only the job),
    /// then records the result and switches transcription to the new
    /// adapter under the lock, then reports it.
    fn work(h: *Home) void {
        const prog: training.Progress = .{ .ctx = h, .onStep = onStep };
        const result = training.run(h.gpa, h.io, &h.model, &h.job.?, h.cfg.training, prog, &h.cancel_flag);
        const ev = blk: {
            h.lock.lockUncancelable(h.io);
            defer h.lock.unlock(h.io);
            var job = h.job.?;
            h.job = null;
            defer job.deinit();
            h.running = false;
            const ev = h.finishLocked(&job, result);
            h.refreshStatusLocked();
            break :blk ev;
        };
        h.emit(ev);
        h.notify();
    }

    fn onStep(ctx: *anyopaque, s: training.Step) void {
        const h: *Home = @ptrCast(@alignCast(ctx));
        const st = blk: {
            h.status_lock.lockUncancelable(h.io);
            defer h.status_lock.unlock(h.io);
            h.status.epoch = @intCast(s.epoch);
            h.status.epochs = @intCast(s.epochs);
            h.status.progress = @as(f32, @floatFromInt(s.done)) / @as(f32, @floatFromInt(@max(s.total, 1)));
            if (h.status.started_unix) |t0| if (s.done > 0) {
                const elapsed = schedule.nowUnix(h.io) - t0;
                h.status.eta_seconds = @divFloor(elapsed * @as(i64, @intCast(s.total - s.done)), @as(i64, @intCast(s.done)));
            };
            if (!std.math.isNan(s.loss)) h.status.last_loss = s.loss;
            break :blk h.status;
        };
        h.emit(.{ .progress = st });
    }

    fn finishLocked(h: *Home, job: *const training.Job, result: anyerror!training.Result) schedule.Event {
        const res = result catch |err| {
            if (err == error.Canceled) {
                h.setIdle();
                return .{ .canceled = h.trainingStatus() };
            }
            h.setError(err);
            return .{ .failed = h.trainingStatus() };
        };
        h.applyLocked(job, res) catch |err| {
            h.setError(err);
            return .{ .failed = h.trainingStatus() };
        };
        return .{ .finished = h.trainingStatus() };
    }

    fn applyLocked(h: *Home, job: *const training.Job, res: training.Result) !void {
        try training.commit(&h.profile, job);
        var ad = try Adapter.load(h.gpa, h.io, job.adapter_path, &h.model);
        defer ad.deinit(h.gpa);
        try h.tr.setAdapter(&ad, res.version);
        // Storage cap: once everything is trained in, the oldest audio can go.
        if ((sync.samplesBytes(&h.profile) catch 0) > h.cfg.storage.max_bytes)
            _ = sync.compact(&h.profile, h.cfg.storage) catch {};
        h.status_lock.lockUncancelable(h.io);
        defer h.status_lock.unlock(h.io);
        h.status.state = .idle;
        h.status.progress = 1;
        h.status.eta_seconds = null;
        h.status.last_finished_unix = schedule.nowUnix(h.io);
        h.status.last_loss = res.loss;
        h.status.last_samples = @intCast(res.samples);
        h.status.last_error = null;
    }

    fn setError(h: *Home, err: anyerror) void {
        h.status_lock.lockUncancelable(h.io);
        defer h.status_lock.unlock(h.io);
        h.status.state = .idle;
        h.status.last_error = @errorName(err);
        h.status.last_finished_unix = schedule.nowUnix(h.io);
    }

    fn setIdle(h: *Home) void {
        h.status_lock.lockUncancelable(h.io);
        defer h.status_lock.unlock(h.io);
        h.status.state = .idle;
    }

    fn refreshStatusLocked(h: *Home) void {
        h.status_lock.lockUncancelable(h.io);
        defer h.status_lock.unlock(h.io);
        const s = &h.status;
        s.pending_samples = training.pending(&h.profile);
        s.min_new_samples = h.cfg.schedule.min_new_samples;
        s.version = h.tr.adapterVersion();
        if (h.running) return;
        s.next_start_unix = null;
        s.state = .idle;
        if (s.pending_samples >= @max(1, s.min_new_samples) and h.profile.learning) {
            if (h.cfg.schedule.window) |w| {
                s.state = .waiting_for_window;
                const wait_min = w.minutesUntilOpen(schedule.localMinute(h.io));
                s.next_start_unix = schedule.nowUnix(h.io) + @as(i64, wait_min) * 60;
            }
        }
    }

    fn emit(h: *Home, ev: schedule.Event) void {
        if (h.listener) |l| l.onEvent(l.ctx, ev);
    }
};

fn nowMs(io: Io) i64 {
    return @intCast(@divFloor(Io.Clock.real.now(io).nanoseconds, std.time.ns_per_ms));
}
