//! Speech transcriber: convolutional audio encoder + autoregressive text
//! decoder (transformer) with KV caching and greedy decoding. The math runs
//! on an accelerator chosen by `Options.backend`:
//!
//!   .cuda   NVIDIA GPU (gpu/engine.zig), the default; the home desktop.
//!   .npu    Snapdragon X Hexagon NPU through QNN (npu/engine.zig); the laptop.
//!   .coreml Core ML (coreml/engine.zig): the iPhone / Mac, Neural Engine first.
//!
//! Both engines expose the same encode/step/pick interface, and this file's
//! decoding loop drives either one. There is no CPU fallback (and no CPU
//! compute threads): `init` fails with error.CudaUnavailable on machines
//! without a usable NVIDIA GPU, or with the NPU engine's error.
const std = @import("std");
const builtin = @import("builtin");
const mel = @import("mel.zig");
const Model = @import("model.zig").Model;
const Tokenizer = @import("bpe.zig").Tokenizer;
const Adapter = @import("adapter.zig").Adapter;
const gpu = @import("gpu/engine.zig");
const npu = @import("npu/engine.zig");
const coreml_engine = @import("coreml/engine.zig");
const Allocator = std.mem.Allocator;

/// Which accelerator runs the model.
pub const Backend = union(enum) {
    /// NVIDIA GPU via the CUDA driver API (home desktop; also used for tests).
    cuda,
    /// Snapdragon X NPU via QNN (the laptop's `stt.Local`).
    npu: npu.Options,
    /// Core ML (iPhone / Mac: Neural Engine, GPU or CPU as Core ML places
    /// each op), from ML Programs generated in Zig (coreml/engine.zig).
    coreml: coreml_engine.Options,
};

pub const Options = struct {
    backend: Backend = .cuda,
    /// Maximum number of text tokens generated per 30 s window.
    max_tokens: usize = 224,
    /// Windows where the model thinks there is no speech are dropped
    /// (thresholds on the no-speech probability and mean log-prob).
    no_speech_threshold: f32 = 0.6,
    logprob_threshold: f32 = -1.0,
    /// Drop outputs that are only an annotation like "[BLANK_AUDIO]" or "(music)".
    drop_annotations: bool = true,
};

/// The engine in use. Every method is the same call on either engine.
pub const Engine = union(enum) {
    gpu: *gpu.Engine,
    npu: *npu.Engine,
    coreml: *coreml_engine.Engine,

    pub fn init(gpa: Allocator, model: *const Model, backend: Backend) !Engine {
        return switch (backend) {
            .cuda => .{ .gpu = try gpu.Engine.init(gpa, model) },
            .npu => |o| .{ .npu = try npu.Engine.init(gpa, model, o) },
            .coreml => |o| .{ .coreml = try coreml_engine.Engine.init(gpa, model, o) },
        };
    }

    pub fn deinit(e: Engine, gpa: Allocator) void {
        switch (e) {
            inline else => |x| x.deinit(gpa),
        }
    }

    pub fn encode(e: Engine, samples: []const f32) !void {
        switch (e) {
            inline else => |x| try x.encode(samples),
        }
    }

    pub fn step(e: Engine, token: u32, pos: u32, want_logits: bool) !void {
        switch (e) {
            inline else => |x| try x.step(token, pos, want_logits),
        }
    }

    pub fn pick(e: Engine, ban_from: u32, ban: [2]u32, nosp: u32) !gpu.Pick {
        return switch (e) {
            inline else => |x| try x.pick(ban_from, ban, nosp),
        };
    }

    pub fn setAdapter(e: Engine, ad: ?*const Adapter, version: u32) !void {
        switch (e) {
            inline else => |x| try x.setAdapter(ad, version),
        }
    }

    pub fn adapterVersion(e: Engine) u32 {
        return switch (e) {
            inline else => |x| x.adapter_version,
        };
    }

    pub fn deviceName(e: Engine) []const u8 {
        return switch (e) {
            inline else => |x| x.deviceName(),
        };
    }
};

pub const Transcriber = struct {
    model: *const Model,
    gpa: Allocator,
    options: Options,
    engine: Engine,
    /// Tokenized vocabulary hint, fed as "previous text" (see `setPrompt`).
    prompt: []u32 = &.{},
    tokenizer: Tokenizer,

    /// Opens the accelerator chosen by `options.backend` and uploads the
    /// weights (GPU: driver JIT on first use, then tens of milliseconds).
    /// `model` must outlive the transcriber.
    pub fn init(gpa: Allocator, model: *const Model, options: Options) !Transcriber {
        var tokenizer = try Tokenizer.init(gpa, model);
        errdefer tokenizer.deinit(gpa);
        const engine = try Engine.init(gpa, model, options.backend);
        return .{ .model = model, .gpa = gpa, .options = options, .engine = engine, .tokenizer = tokenizer };
    }

    pub fn deinit(self: *Transcriber) void {
        self.engine.deinit(self.gpa);
        self.gpa.free(self.prompt);
        self.tokenizer.deinit(self.gpa);
        self.* = undefined;
    }

    /// Merges a personal adapter into the engine's weights (or, with null,
    /// goes back to the base model). `version` is recorded as `adapterVersion()`.
    pub fn setAdapter(self: *Transcriber, ad: ?*const Adapter, version: u32) !void {
        try self.engine.setAdapter(ad, version);
    }

    pub fn adapterVersion(self: *const Transcriber) u32 {
        return self.engine.adapterVersion();
    }

    /// The accelerator's name (GPU model, or "Hexagon NPU").
    pub fn deviceName(self: *const Transcriber) []const u8 {
        return self.engine.deviceName();
    }

    /// GPU timings for the last window, and the one-time weight upload (ms).
    /// All zero on the NPU backend.
    pub fn gpuTiming(self: *const Transcriber) struct { stages: gpu.Timing, upload_ms: f32, weight_mb: usize } {
        const e = switch (self.engine) {
            .gpu => |g| g,
            .npu, .coreml => return .{ .stages = .{}, .upload_ms = 0, .weight_mb = 0 },
        };
        return .{ .stages = e.timings(), .upload_ms = e.upload_ms, .weight_mb = e.weight_bytes >> 20 };
    }

    /// Sets a text hint that biases spelling and style, e.g. names and jargon
    /// you dictate often ("Zig, comptime, allocator"). Pass "" to clear.
    pub fn setPrompt(self: *Transcriber, text: []const u8) !void {
        self.gpa.free(self.prompt);
        self.prompt = &.{};
        const trimmed = std.mem.trim(u8, text, " ");
        if (trimmed.len == 0) return;
        const spaced = try std.mem.concat(self.gpa, u8, &.{ " ", trimmed });
        defer self.gpa.free(spaced);
        const toks = try self.tokenizer.encode(self.gpa, spaced);
        defer self.gpa.free(toks);
        // Previous text may use at most half the decoder context.
        const keep = @min(toks.len, self.model.hp.n_text_ctx / 2 - 1);
        self.prompt = try self.gpa.dupe(u32, toks[toks.len - keep ..]);
    }

    /// Transcribes 16 kHz mono f32 samples (any length) to UTF-8 text.
    /// Audio longer than 30 s is split at quiet points. Caller owns the result.
    pub fn transcribe(self: *Transcriber, gpa: Allocator, samples: []const f32) ![]u8 {
        var text: std.ArrayList(u8) = .empty;
        errdefer text.deinit(gpa);
        var start: usize = 0;
        while (start < samples.len) {
            const end = splitPoint(samples, start);
            // Skip windows too short to contain a word (< 0.1 s).
            if (end - start >= mel.sample_rate / 10) {
                const part = try self.transcribeWindow(gpa, samples[start..end]);
                defer gpa.free(part);
                const trimmed = std.mem.trim(u8, part, " \t\r\n");
                if (trimmed.len > 0) {
                    if (text.items.len > 0) try text.append(gpa, ' ');
                    try text.appendSlice(gpa, trimmed);
                }
            }
            start = end;
        }
        return text.toOwnedSlice(gpa);
    }

    /// Transcribes at most 30 s of audio: encode on the accelerator, then greedy decoding.
    fn transcribeWindow(self: *Transcriber, gpa: Allocator, samples: []const f32) ![]u8 {
        const e = self.engine;
        const m = self.model;
        const hp = m.hp;
        const tk = m.tokens;
        try e.encode(samples); // the engine computes the spectrogram too

        var prompt: std.ArrayList(u32) = .empty;
        defer prompt.deinit(gpa);
        if (self.prompt.len > 0) {
            try prompt.append(gpa, tk.prev);
            try prompt.appendSlice(gpa, self.prompt);
        }
        const sot_index = prompt.items.len;
        try prompt.append(gpa, tk.sot);
        if (tk.multilingual) {
            try prompt.append(gpa, tk.lang_en);
            try prompt.append(gpa, tk.transcribe);
        }
        try prompt.append(gpa, tk.not);

        const n_vocab: u32 = @intCast(hp.n_vocab);
        var pos: u32 = 0;
        var no_speech_prob: f32 = 0;
        for (prompt.items, 0..) |tok, i| {
            const last = i == prompt.items.len - 1;
            try e.step(tok, pos, i == sot_index or last);
            pos += 1;
            if (i == sot_index) no_speech_prob = @exp((try e.pick(n_vocab, .{ none, none }, tk.nosp)).nosp_logprob);
        }

        var out: std.ArrayList(u32) = .empty;
        defer out.deinit(gpa);
        var sum_logprob: f32 = 0;
        const limit = @min(self.options.max_tokens, hp.n_text_ctx - pos);
        while (out.items.len < limit) {
            // Only plain text tokens (and end-of-text) are allowed; the first
            // token may be neither end-of-text nor a lone space (id 220).
            const ban: [2]u32 = if (out.items.len == 0) .{ tk.eot, 220 } else .{ none, none };
            const p = try e.pick(tk.eot + 1, ban, tk.nosp);
            const best = p.token;
            sum_logprob += p.logprob;
            if (best == tk.eot) break;
            try out.append(gpa, best);
            if (loopLength(out.items)) |n| {
                // Keep one copy of the repeated phrase, drop the rest.
                out.shrinkRetainingCapacity(out.items.len - n.len * (n.reps - 1));
                break;
            }
            if (out.items.len >= limit) break; // no logits needed after the last token
            try e.step(best, pos, true);
            pos += 1;
        }

        const n_tok: f32 = @floatFromInt(@max(out.items.len, 1));
        if (no_speech_prob > self.options.no_speech_threshold and
            sum_logprob / n_tok < self.options.logprob_threshold)
        {
            return gpa.alloc(u8, 0);
        }

        var text: std.ArrayList(u8) = .empty;
        errdefer text.deinit(gpa);
        for (out.items) |t| {
            if (t < m.vocab.len) try text.appendSlice(gpa, m.vocab[t]);
        }
        if (self.options.drop_annotations and isAnnotation(text.items)) text.clearRetainingCapacity();
        return text.toOwnedSlice(gpa);
    }
};

const none = gpu.Engine.none;

/// True for outputs like " [BLANK_AUDIO]", "(music)" or "*silence*".
fn isAnnotation(text: []const u8) bool {
    const t = std.mem.trim(u8, text, " .\t\r\n");
    if (t.len < 2) return false;
    const pairs = [_][2]u8{ .{ '[', ']' }, .{ '(', ')' }, .{ '*', '*' } };
    for (pairs) |p| if (t[0] == p[0] and t[t.len - 1] == p[1]) return true;
    return false;
}

const Loop = struct { len: usize, reps: usize };

/// Detects degenerate output where a short n-gram (up to 8 tokens) repeats
/// back-to-back 4 times. Longer phrases are left alone: a speaker really can
/// say the same sentence several times, and dropping real speech is worse.
fn loopLength(toks: []const u32) ?Loop {
    const reps = 4;
    var n: usize = 1;
    while (n <= 8) : (n += 1) {
        if (toks.len < n * reps) break;
        const tail = toks[toks.len - n ..];
        var ok = true;
        for (1..reps) |r| {
            const seg = toks[toks.len - n * (r + 1) ..][0..n];
            if (!std.mem.eql(u32, seg, tail)) {
                ok = false;
                break;
            }
        }
        if (ok) return .{ .len = n, .reps = reps };
    }
    return null;
}

/// Picks where the window starting at `start` should end: 30 s later, or
/// earlier at the quietest 20 ms in the last 2 s so words aren't cut in half.
fn splitPoint(samples: []const f32, start: usize) usize {
    const max_end = start + mel.n_samples;
    if (max_end >= samples.len) return samples.len;
    const frame = mel.sample_rate / 50;
    const search = 2 * mel.sample_rate;
    var best_end = max_end;
    var best_energy: f32 = std.math.inf(f32);
    var e = max_end;
    while (e >= max_end - search + frame) : (e -= frame) {
        var energy: f32 = 0;
        for (samples[e - frame .. e]) |s| energy += s * s;
        if (energy < best_energy) {
            best_energy = energy;
            best_end = e;
        }
    }
    return best_end;
}

test "loopLength" {
    try std.testing.expectEqual(Loop{ .len = 2, .reps = 4 }, loopLength(&.{ 1, 2, 1, 2, 1, 2, 1, 2 }).?);
    try std.testing.expect(loopLength(&.{ 1, 2, 3, 4, 5 }) == null);
    try std.testing.expect(loopLength(&.{ 7, 7, 7 }) == null); // "no no no" is fine
    const phrase = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    try std.testing.expect(loopLength(&(phrase ++ phrase ++ phrase)) == null); // a repeated sentence is real speech
}
