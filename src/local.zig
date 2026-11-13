//! The laptop: speech-to-text that runs locally on the laptop's NPU
//! (Snapdragon X Hexagon, through QNN). Laptop audio never leaves the laptop;
//! only the training samples made from your feedback go home, as sync
//! bundles, and adapter updates come back (`importUpdate`).
//!
//! Portal captures the mic and hands `Local` 16 kHz mono PCM (or audio file
//! bytes), relays the three feedback responses, and moves the bundles and
//! updates. The `stt-local` executable (local_main.zig) wraps this object in
//! a JSON-lines protocol for Portal.
//!
//! Event-driven, no polling: the only timed work is the silence
//! auto-confirm. `tick` returns when it next needs to run and the wakeup
//! hook reports whenever that deadline changes (same semantics as `Home`).
//!
//! All methods are thread-safe (serialized by one lock). Local never starts
//! threads. It runs from a read-only install folder: the model and the
//! optional adapter package are only read, and everything written (samples,
//! rules, vocabulary, the adapter from updates, the NPU's compiled-graph
//! cache) goes under `profile_dir`.
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
const audio = @import("audio.zig");
const Adapter = @import("adapter.zig").Adapter;
const npu = @import("npu/engine.zig");
const home = @import("home.zig");

pub const Result = service.Result;
pub const Outcome = service.Outcome;
pub const Bundle = sync.Bundle;
/// Same hook as `Home`'s: told the unix-ms time `tick` should next run
/// (null = nothing pending). Don't call back into `Local` from it.
pub const Wakeup = home.Wakeup;

/// Which accelerator runs the model.
pub const Backend = enum {
    /// The laptop's NPU (the real configuration). `npu` options apply.
    npu,
    /// An NVIDIA GPU: for testing the laptop flow on a desktop.
    cuda,
};

pub const Config = struct {
    /// The base model (the `stt-model` data package). Read only.
    model_path: []const u8,
    /// The only folder Local writes to (Portal's data dir for stt-local).
    profile_dir: []const u8,
    /// Optional adapter shipped as the `stt-adapter` data package. Read only.
    /// Either a raw adapter file (`adapter.bin`, STTLORA1) or an update
    /// (STTUPD01, as from `Home.exportUpdate`). A raw adapter is used when
    /// the profile has none of its own or `adapter_version` is newer than
    /// the profile's; an update is imported into the profile like
    /// `importUpdate` (if newer).
    adapter_path: ?[]const u8 = null,
    /// Version of a raw adapter at `adapter_path` (0 = unknown/oldest).
    adapter_version: u32 = 0,
    backend: Backend = .npu,
    /// NPU options. `cache_dir` = null means `profile_dir`.
    npu: npu.Options = .{},
    /// Name this device's samples carry in sync bundles.
    device_name: []const u8 = "laptop",
    service: service.Options = .{},
    /// Decoding options (its `backend` field is set from `backend`/`npu`).
    transcriber: transcriber.Options = .{},
    /// Vocabulary hint added in front of the learned words.
    base_prompt: []const u8 = "",
};

pub const Status = struct {
    backend: Backend,
    /// The accelerator's name ("Hexagon NPU", or the GPU's name).
    device: []const u8,
    /// Adapter version transcription is using (0 = base model).
    adapter_version: u32,
    /// Adapter version recorded in the profile (what to tell home on sync).
    profile_version: u32,
    learning: bool,
    /// Samples stored locally and not yet delivered home.
    pending_export: u32,
    /// When `tick` next has work (unix ms), or null.
    deadline_unix_ms: ?i64,
};

pub const Local = struct {
    gpa: Allocator,
    io: Io,
    cfg: Config,
    lock: Io.Mutex = .init,
    model: Model,
    profile: Profile,
    tr: Transcriber,
    svc: Service,
    wakeup: ?Wakeup = null,
    /// Set once the engine has failed; every later transcription returns
    /// error.BackendLost (the host should restart).
    lost: bool = false,

    pub fn open(gpa: Allocator, io: Io, cfg: Config) !*Local {
        const l = try gpa.create(Local);
        errdefer gpa.destroy(l);
        l.* = .{ .gpa = gpa, .io = io, .cfg = cfg, .model = undefined, .profile = undefined, .tr = undefined, .svc = undefined };
        l.model = try Model.load(gpa, io, cfg.model_path);
        errdefer l.model.deinit();
        l.profile = try Profile.load(gpa, io, cfg.profile_dir);
        errdefer l.profile.deinit();

        var topts = cfg.transcriber;
        topts.backend = switch (cfg.backend) {
            .cuda => .cuda,
            .npu => blk: {
                var o = cfg.npu;
                if (o.cache_dir == null) o.cache_dir = cfg.profile_dir;
                break :blk .{ .npu = o };
            },
        };
        l.tr = try Transcriber.init(gpa, &l.model, topts);
        errdefer l.tr.deinit();
        try l.loadAdapters();
        try l.refreshPromptLocked();
        l.svc = try Service.init(gpa, io, &l.tr, &l.profile, cfg.service);
        return l;
    }

    /// Confirms pending transcripts (silence = correct) and frees everything.
    pub fn close(l: *Local) void {
        l.svc.deinit();
        l.tr.deinit();
        l.profile.deinit();
        l.model.deinit();
        l.gpa.destroy(l);
    }

    /// Sets the wakeup hook and reports the current deadline right away.
    pub fn setWakeup(l: *Local, w: ?Wakeup) void {
        l.lock.lockUncancelable(l.io);
        l.wakeup = w;
        l.lock.unlock(l.io);
        l.notify();
    }

    // ------------------------------------------------------------ transcription

    /// Transcribes an audio file's bytes (WAV, MP3 or QOA; any rate/channels).
    /// Errors: BadAudio, AudioTooLong, BackendLost, OutOfMemory.
    /// Caller owns `text`.
    pub fn transcribe(l: *Local, gpa: Allocator, audio_bytes: []const u8) !Result {
        const samples = audio.decode(gpa, audio_bytes, .{ .max_seconds = l.cfg.service.max_seconds }) catch |err| return switch (err) {
            error.AudioTooLong, error.OutOfMemory => err,
            else => error.BadAudio,
        };
        defer gpa.free(samples);
        return l.transcribePcm(gpa, samples);
    }

    /// Transcribes 16 kHz mono f32 samples. Caller owns `text`.
    pub fn transcribePcm(l: *Local, gpa: Allocator, samples: []const f32) !Result {
        const r = blk: {
            l.lock.lockUncancelable(l.io);
            defer l.lock.unlock(l.io);
            if (l.lost) return error.BackendLost;
            break :blk l.svc.transcribe(gpa, samples, "") catch |err| switch (err) {
                error.AudioTooLong, error.OutOfMemory => return err,
                else => {
                    std.log.err("stt-local: engine failed: {t}", .{err});
                    l.lost = true;
                    return error.BackendLost;
                },
            };
        };
        l.notify(); // a new transcript is waiting for feedback / silence
        return r;
    }

    /// One of the three responses for transcript `id`: fix (corrected text,
    /// true), discard (_, false). Silence needs no call.
    pub fn feedback(l: *Local, id: u64, text: []const u8, success: bool) !Outcome {
        const out = blk: {
            l.lock.lockUncancelable(l.io);
            defer l.lock.unlock(l.io);
            break :blk try l.svc.feedback(id, text, success);
        };
        l.notify();
        return out;
    }

    /// "nothing": stop keeping audio and learning (or resume).
    pub fn setLearning(l: *Local, on: bool) !void {
        l.lock.lockUncancelable(l.io);
        defer l.lock.unlock(l.io);
        try l.profile.setLearning(on);
    }

    // ------------------------------------------------------------ sync (device side)

    /// Packs pending samples (oldest first, at most `max_bytes`, always at
    /// least one) for home's `importBundle`. null = nothing pending (or
    /// learning off). Call `commitExport(b.through)` once home stored it.
    pub fn exportBundle(l: *Local, gpa: Allocator, max_bytes: usize) !?Bundle {
        l.lock.lockUncancelable(l.io);
        defer l.lock.unlock(l.io);
        return sync.exportBundle(gpa, &l.profile, l.cfg.device_name, max_bytes);
    }

    /// Deletes the samples home now holds (ids up to `through`).
    pub fn commitExport(l: *Local, through: u32) !void {
        l.lock.lockUncancelable(l.io);
        defer l.lock.unlock(l.io);
        try sync.commitExport(&l.profile, through);
    }

    /// Installs an update from home (`Home.exportUpdate`): the adapter is
    /// stored in the profile and switched into the engine right away, and
    /// the rules and vocabulary apply from the next transcription. Returns
    /// the version now installed (unchanged if the update isn't newer).
    pub fn importUpdate(l: *Local, gpa: Allocator, bytes: []const u8) !u32 {
        l.lock.lockUncancelable(l.io);
        defer l.lock.unlock(l.io);
        const before = l.profile.version;
        const v = try sync.importUpdate(gpa, &l.profile, bytes);
        if (v != before) {
            if (try training.loadAdapter(l.gpa, &l.profile, &l.model)) |ad_| {
                var ad = ad_;
                defer ad.deinit(l.gpa);
                try l.tr.setAdapter(&ad, v);
            }
            try l.refreshPromptLocked();
        }
        return v;
    }

    // ------------------------------------------------------------ status / timing

    pub fn status(l: *Local) Status {
        l.lock.lockUncancelable(l.io);
        defer l.lock.unlock(l.io);
        return .{
            .backend = l.cfg.backend,
            .device = l.tr.deviceName(),
            .adapter_version = l.tr.adapterVersion(),
            .profile_version = l.profile.version,
            .learning = l.profile.learning,
            .pending_export = (l.profile.next_sample - 1) -| l.profile.exported,
            .deadline_unix_ms = l.deadlineLocked(),
        };
    }

    /// Adapter version transcription is using.
    pub fn adapterVersion(l: *Local) u32 {
        l.lock.lockUncancelable(l.io);
        defer l.lock.unlock(l.io);
        return l.tr.adapterVersion();
    }

    /// Confirms transcripts whose silence period has passed. Returns the
    /// unix-ms time it next needs to run, or null if nothing is pending (the
    /// wakeup hook reports later changes). Calling it early is harmless.
    pub fn tick(l: *Local) ?i64 {
        l.lock.lockUncancelable(l.io);
        defer l.lock.unlock(l.io);
        l.svc.tick();
        return l.deadlineLocked();
    }

    fn deadlineLocked(l: *Local) ?i64 {
        const ms = l.svc.msUntilTick() orelse return null;
        return nowMs(l.io) + ms;
    }

    fn notify(l: *Local) void {
        const w, const deadline = blk: {
            l.lock.lockUncancelable(l.io);
            defer l.lock.unlock(l.io);
            break :blk .{ l.wakeup orelse return, l.deadlineLocked() };
        };
        w.onWakeup(w.ctx, deadline);
    }

    fn refreshPromptLocked(l: *Local) !void {
        const pt = try l.profile.promptText(l.gpa, l.cfg.base_prompt);
        defer l.gpa.free(pt);
        try l.tr.setPrompt(pt);
    }

    /// Picks the adapter at open: the profile's own (from earlier updates)
    /// or the read-only package at `adapter_path`, whichever is newer.
    fn loadAdapters(l: *Local) !void {
        const gpa = l.gpa;
        var pkg: ?Adapter = null;
        defer if (pkg) |*a| a.deinit(gpa);
        if (l.cfg.adapter_path) |path| {
            const bytes = try Io.Dir.cwd().readFileAlloc(l.io, path, gpa, .limited(1 << 30));
            defer gpa.free(bytes);
            if (std.mem.startsWith(u8, bytes, "STTUPD01")) {
                _ = try sync.importUpdate(gpa, &l.profile, bytes);
            } else if (l.cfg.adapter_version > l.profile.version or !try l.profileHasAdapter()) {
                pkg = try Adapter.parse(gpa, bytes, &l.model);
            }
        }
        if (pkg) |*a| return l.tr.setAdapter(a, l.cfg.adapter_version);
        if (try training.loadAdapter(gpa, &l.profile, &l.model)) |ad_| {
            var ad = ad_;
            defer ad.deinit(gpa);
            try l.tr.setAdapter(&ad, l.profile.version);
        }
    }

    fn profileHasAdapter(l: *Local) !bool {
        Io.Dir.cwd().access(l.io, try l.profile.path("adapter.bin"), .{}) catch return false;
        return true;
    }
};

fn nowMs(io: Io) i64 {
    return @intCast(@divFloor(Io.Clock.real.now(io).nanoseconds, std.time.ns_per_ms));
}
