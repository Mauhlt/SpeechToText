//! When home training runs, and what it is doing right now.
//!
//! Training uses the GPU heavily for minutes at a time, which slows down
//! anything else using the GPU (transcription included). So it only *starts*
//! inside a daily time window (local time, default 02:00–06:00) once enough
//! new samples have arrived, or when asked explicitly (`Home.trainNow`).
//! The host (Portal) reads `Status` / receives `Event`s to warn the user
//! while a run is in progress.
const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

/// Daily window in local time, as minutes after midnight. `start == end`
/// means "any time". The window may wrap midnight (e.g. 22:00–06:00).
pub const Window = struct {
    start_minute: u16 = 2 * 60,
    end_minute: u16 = 6 * 60,

    pub fn contains(w: Window, minute: u16) bool {
        if (w.start_minute == w.end_minute) return true;
        if (w.start_minute < w.end_minute) return minute >= w.start_minute and minute < w.end_minute;
        return minute >= w.start_minute or minute < w.end_minute;
    }

    /// Minutes from `minute` until the window next opens (0 if open now).
    pub fn minutesUntilOpen(w: Window, minute: u16) u16 {
        if (w.contains(minute)) return 0;
        return @intCast((@as(u32, w.start_minute) + 24 * 60 - minute) % (24 * 60));
    }
};

pub const Config = struct {
    /// null = never start automatically (only `trainNow`).
    window: ?Window = .{},
    /// New samples needed since the last version before training again.
    min_new_samples: u32 = 20,
};

pub const State = enum {
    /// Not enough new samples yet.
    idle,
    /// Enough samples; waiting for the time window to open.
    waiting_for_window,
    /// Training now: expect the GPU (and transcription) to be slower.
    running,
};

pub const Status = struct {
    state: State = .idle,
    /// Samples added since the last trained version, and how many are needed.
    pending_samples: u32 = 0,
    min_new_samples: u32 = 20,
    /// Unix time (s) when the window next opens (state = waiting_for_window).
    next_start_unix: ?i64 = null,
    /// Current run (state = running).
    started_unix: ?i64 = null,
    /// 0..1 through the current run.
    progress: f32 = 0,
    /// Estimated seconds left in the current run (after the first step).
    eta_seconds: ?i64 = null,
    epoch: u32 = 0,
    epochs: u32 = 0,
    /// Adapter version transcription is using.
    version: u32 = 0,
    /// Previous run.
    last_finished_unix: ?i64 = null,
    last_loss: ?f32 = null,
    last_samples: u32 = 0,
    /// Name of the error that ended the previous run, if it failed.
    last_error: ?[]const u8 = null,
};

pub const Event = union(enum) {
    /// Training started: warn the user other GPU work will be slower.
    started: Status,
    /// After every optimizer step (progress, eta).
    progress: Status,
    /// New adapter version trained and already in use for transcription.
    finished: Status,
    /// The run stopped early; the previous version stays in use.
    failed: Status,
    canceled: Status,
};

/// Receives training events: `started` from whichever call started the run
/// (`tick` or `trainNow`), everything else from the training thread. Keep it
/// quick and don't call back into `Home` from it.
pub const Listener = struct {
    ctx: *anyopaque,
    onEvent: *const fn (ctx: *anyopaque, ev: Event) void,
};

/// Unix time in seconds.
pub fn nowUnix(io: Io) i64 {
    return @intCast(@divFloor(Io.Clock.real.now(io).nanoseconds, std.time.ns_per_s));
}

const SYSTEMTIME = extern struct {
    year: u16,
    month: u16,
    day_of_week: u16,
    day: u16,
    hour: u16,
    minute: u16,
    second: u16,
    ms: u16,
};
extern "kernel32" fn GetLocalTime(t: *SYSTEMTIME) callconv(.winapi) void;

/// Minutes after local midnight. Windows reads the OS local time; elsewhere
/// this is UTC (set a window in UTC, or `window = null` and call trainNow).
pub fn localMinute(io: Io) u16 {
    if (builtin.os.tag == .windows) {
        var t: SYSTEMTIME = undefined;
        GetLocalTime(&t);
        return t.hour * 60 + t.minute;
    }
    const day_s: i64 = @mod(nowUnix(io), 24 * 3600);
    return @intCast(@divFloor(day_s, 60));
}

test "window" {
    const w: Window = .{ .start_minute = 120, .end_minute = 360 };
    try std.testing.expect(w.contains(120) and w.contains(359) and !w.contains(360) and !w.contains(0));
    try std.testing.expectEqual(@as(u16, 60), w.minutesUntilOpen(60));
    try std.testing.expectEqual(@as(u16, 24 * 60 - 240), w.minutesUntilOpen(360));
    const night: Window = .{ .start_minute = 22 * 60, .end_minute = 6 * 60 };
    try std.testing.expect(night.contains(23 * 60) and night.contains(60) and !night.contains(12 * 60));
    const any: Window = .{ .start_minute = 0, .end_minute = 0 };
    try std.testing.expect(any.contains(777));
}
