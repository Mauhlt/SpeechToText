//! One round of home-side training: turn the profile's samples into the next
//! adapter version, on the GPU.
//!
//! Split in three so the slow part can run on a background thread:
//!   collect(profile)  -> Job      reads the profile (caller holds its lock)
//!   run(job)          -> Result   GPU training; touches only the job's files
//!   commit(profile)              records the new version (caller holds the lock)
//!
//! Sample files are immutable once their .txt exists, and adapter files are
//! replaced by atomic rename, so `run` can read and write them while the
//! profile keeps taking new samples.
const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const Model = @import("model.zig").Model;
const Profile = @import("profile.zig").Profile;
const Adapter = @import("adapter.zig").Adapter;
const trainer = @import("gpu/trainer.zig");
const audio = @import("audio.zig");

pub const Options = trainer.Options;
pub const Progress = trainer.Progress;
pub const Step = trainer.Step;

pub const Job = struct {
    /// Heap-allocated so the arena never moves (the struct is returned by value).
    arena: *std.heap.ArenaAllocator,
    /// Sample files to train on (wav, txt), oldest first.
    samples: []const [2][]const u8,
    /// Newest sample id included; becomes `profile.trained` on commit.
    newest: u32,
    /// The version this job produces.
    version: u32,
    adapter_path: []const u8,
    archive_path: []const u8,
    archive_dir: []const u8,

    pub fn deinit(j: *Job) void {
        const gpa = j.arena.child_allocator;
        j.arena.deinit();
        gpa.destroy(j.arena);
        j.* = undefined;
    }
};

pub const Result = struct {
    version: u32,
    samples: usize,
    loss: f32,
};

/// Samples added since the last trained version.
pub fn pending(p: *const Profile) u32 {
    const newest = p.next_sample - 1;
    return newest -| p.trained;
}

/// Lists every sample the profile holds (training revisits old samples too,
/// so the adapter doesn't forget them). Caller holds the profile's lock.
pub fn collect(gpa: Allocator, p: *Profile) !Job {
    const arena = try gpa.create(std.heap.ArenaAllocator);
    errdefer gpa.destroy(arena);
    arena.* = .init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    var list: std.ArrayList([2][]const u8) = .empty;
    var name: [48]u8 = undefined;
    const cwd = Io.Dir.cwd();
    var id: u32 = 1;
    while (id < p.next_sample) : (id += 1) {
        const txt = try std.fs.path.join(a, &.{ p.dir, try std.fmt.bufPrint(&name, "samples/{d:0>6}.txt", .{id}) });
        const wav = try std.fs.path.join(a, &.{ p.dir, try std.fmt.bufPrint(&name, "samples/{d:0>6}.wav", .{id}) });
        // Skip ids whose audio was compacted away or never completed.
        _ = cwd.statFile(p.io, txt, .{}) catch continue;
        _ = cwd.statFile(p.io, wav, .{}) catch continue;
        try list.append(a, .{ wav, txt });
    }
    const version = p.version + 1;
    return .{
        .arena = arena,
        .samples = list.items,
        .newest = p.next_sample - 1,
        .version = version,
        .adapter_path = try std.fs.path.join(a, &.{ p.dir, "adapter.bin" }),
        .archive_dir = try std.fs.path.join(a, &.{ p.dir, "adapters" }),
        .archive_path = try std.fs.path.join(a, &.{ p.dir, "adapters", try std.fmt.bufPrint(&name, "v{d:0>4}.bin", .{version}) }),
    };
}

/// Trains (or continues training) the adapter on the job's samples and
/// writes `adapter.bin` plus an archived copy `adapters/vNNNN.bin`. Safe to
/// call on a background thread: `model` is only read.
pub fn run(gpa: Allocator, io: Io, model: *const Model, job: *const Job, opts: Options, progress: ?Progress, cancel: ?*const std.atomic.Value(bool)) !Result {
    if (job.samples.len == 0) return error.NoSamples;
    var auds: std.ArrayList([]f32) = .empty;
    defer {
        for (auds.items) |x| gpa.free(x);
        auds.deinit(gpa);
    }
    var texts: std.ArrayList([]u8) = .empty;
    defer {
        for (texts.items) |x| gpa.free(x);
        texts.deinit(gpa);
    }
    const cwd = Io.Dir.cwd();
    for (job.samples) |f| {
        const t = cwd.readFileAlloc(io, f[1], gpa, .limited(1 << 20)) catch continue;
        const a = audio.load(gpa, io, f[0], .{ .max_seconds = 31 }) catch {
            gpa.free(t);
            continue;
        };
        try texts.append(gpa, t);
        try auds.append(gpa, a);
    }
    if (auds.items.len == 0) return error.NoSamples;
    const samples = try gpa.alloc(trainer.Sample, auds.items.len);
    defer gpa.free(samples);
    for (samples, auds.items, texts.items) |*s, a, t| s.* = .{ .audio = a, .text = t };

    // Start fresh only if there is no adapter yet. Any other failure (corrupt
    // file, different base model) must not silently overwrite trained weights.
    var ad = Adapter.load(gpa, io, job.adapter_path, model) catch |err| switch (err) {
        error.FileNotFound => try Adapter.init(gpa, model, Adapter.default_rank, Adapter.default_alpha, opts.seed),
        else => return err,
    };
    defer ad.deinit(gpa);
    const loss = try trainer.train(gpa, model, &ad, samples, opts, progress, cancel);

    try cwd.createDirPath(io, job.archive_dir);
    try ad.save(gpa, io, job.archive_path);
    try ad.save(gpa, io, job.adapter_path);
    return .{ .version = job.version, .samples = samples.len, .loss = loss };
}

/// Records a finished job in the profile. Caller holds the profile's lock.
pub fn commit(p: *Profile, job: *const Job) !void {
    p.version = job.version;
    p.trained = @max(p.trained, job.newest);
    try p.saveState();
}

/// The profile's current adapter, or null if none has been trained yet.
pub fn loadAdapter(gpa: Allocator, p: *Profile, model: *const Model) !?Adapter {
    return Adapter.load(gpa, p.io, try p.path("adapter.bin"), model) catch |err| switch (err) {
        error.FileNotFound => null,
        else => err,
    };
}
