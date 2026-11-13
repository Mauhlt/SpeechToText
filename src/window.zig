//! Feedback window: recent transcripts kept (with their audio) until the
//! host responds. Used by `Service`.
//!
//!   fix      feedback(id, corrected_text, true)  -> learned
//!   silence  no response within `confirm_after_ms`, eviction by newer
//!            transcripts, or shutdown            -> saved as correct
//!   nothing  profile learning off                 -> audio never kept
//!
//! Not thread-safe; callers serialize access.
const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const profile_mod = @import("profile.zig");
const Profile = profile_mod.Profile;

pub const Outcome = union(enum) {
    /// Stored as training data (and rules/vocab updated if it was a correction).
    learned: profile_mod.LearnResult,
    /// success = false: nothing was stored.
    discarded,
    /// The id is unknown or has aged out of the window.
    unknown_id,
    no_profile,
    failed: anyerror,
};

const Recent = struct {
    id: u64,
    audio: []f32,
    text: []u8,
    source: []u8,
    at: Io.Timestamp,
};

pub const Window = struct {
    gpa: Allocator,
    io: Io,
    profile: ?*Profile,
    confirm_after_ms: i64,
    slots: []?Recent,
    next_id: u64 = 1,

    pub fn init(gpa: Allocator, io: Io, profile: ?*Profile, capacity: usize, confirm_after_ms: i64) !Window {
        const slots = try gpa.alloc(?Recent, @max(capacity, 1));
        @memset(slots, null);
        return .{ .gpa = gpa, .io = io, .profile = profile, .confirm_after_ms = confirm_after_ms, .slots = slots };
    }

    /// Drops everything still pending without saving it. Call `confirmAll`
    /// first for the normal "shutdown = silence = correct" behavior.
    pub fn deinit(w: *Window) void {
        for (w.slots) |*s| w.drop(s);
        w.gpa.free(w.slots);
        w.* = undefined;
    }

    /// Registers a delivered transcript and returns its id. The audio is
    /// copied only if the profile is learning. `source` names the device
    /// the audio came from ("" = this one).
    pub fn add(w: *Window, audio: []const f32, text: []const u8, source: []const u8) !u64 {
        const id = w.next_id;
        w.next_id += 1;
        const slot = &w.slots[id % w.slots.len];
        w.confirm(slot); // the oldest one ages out: silence = correct
        const p = w.profile orelse return id;
        if (!p.learning) return id;
        const a = try w.gpa.dupe(f32, audio);
        errdefer w.gpa.free(a);
        const t = try w.gpa.dupe(u8, text);
        errdefer w.gpa.free(t);
        slot.* = .{ .id = id, .audio = a, .text = t, .source = try w.gpa.dupe(u8, source), .at = Io.Clock.awake.now(w.io) };
        return id;
    }

    /// Applies the host's response to transcript `id`.
    pub fn feedback(w: *Window, id: u64, text: []const u8, success: bool) Outcome {
        const p = w.profile orelse return .no_profile;
        const slot = &w.slots[id % w.slots.len];
        const rec = slot.* orelse return .unknown_id;
        if (rec.id != id) return .unknown_id;
        defer w.drop(slot);
        const res = p.feedbackFrom(w.gpa, rec.audio, rec.text, .{ .text = text, .success = success }, rec.source) catch |e|
            return .{ .failed = e };
        if (res.sample_id == null and res.new_rules == 0 and res.new_words == 0) return .discarded;
        return .{ .learned = res };
    }

    /// Confirms (saves as correct) everything older than `confirm_after_ms`.
    pub fn confirmExpired(w: *Window) void {
        const now = Io.Clock.awake.now(w.io);
        for (w.slots) |*s| if (s.*) |rec| {
            if (rec.at.durationTo(now).toMilliseconds() >= w.confirm_after_ms) w.confirm(s);
        };
    }

    /// Milliseconds until the oldest pending transcript auto-confirms (0 if
    /// overdue), or null if nothing is pending.
    pub fn msUntilNextConfirm(w: *const Window) ?i64 {
        const now = Io.Clock.awake.now(w.io);
        var best: ?i64 = null;
        for (w.slots) |s| if (s) |rec| {
            const left = @max(0, w.confirm_after_ms - rec.at.durationTo(now).toMilliseconds());
            best = if (best) |b| @min(b, left) else left;
        };
        return best;
    }

    /// Confirms everything still pending (e.g. at shutdown).
    pub fn confirmAll(w: *Window) void {
        for (w.slots) |*s| w.confirm(s);
    }

    fn confirm(w: *Window, slot: *?Recent) void {
        const rec = slot.* orelse return;
        defer w.drop(slot);
        const p = w.profile orelse return;
        _ = p.feedbackFrom(w.gpa, rec.audio, rec.text, .{ .text = rec.text, .success = true }, rec.source) catch |e|
            std.log.warn("stt: could not save confirmed sample {d}: {t}", .{ rec.id, e });
    }

    fn drop(w: *Window, slot: *?Recent) void {
        if (slot.*) |r| {
            w.gpa.free(r.audio);
            w.gpa.free(r.text);
            w.gpa.free(r.source);
        }
        slot.* = null;
    }
};
