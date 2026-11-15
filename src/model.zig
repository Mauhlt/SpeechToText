//! Loader for speech-model weights stored in the ggml `.bin` container format.
//! (The file is only data: hyperparameters, mel filters, vocab, tensors.)
//!
//! Layout:
//!   u32 magic 'ggml'
//!   11 x i32 hparams
//!   i32 n_mel, i32 n_fft, f32[n_mel*n_fft] mel filterbank
//!   i32 n_vocab, then n_vocab x (u32 len, bytes)
//!   tensors until EOF: i32 n_dims, i32 name_len, i32 type,
//!                      i32[n_dims] dims (innermost first), name, data
//!
//! The host copy is only a staging area for the GPU: weight matrices stay f16
//! (as stored in the file, and as the GPU uses them); small vectors (biases,
//! norms, positional tables) are f32. The host copy is never modified, so one
//! `Model` can back an inference engine and a training engine at once.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// A dense layer: y = x · Wᵀ + b, with W stored row-major as [n_out][n_in]
/// (PyTorch nn.Linear layout).
pub const Linear = struct {
    w: []const f16,
    b: ?[]const f32,
    n_in: usize,
    n_out: usize,
};

pub const HParams = struct {
    n_vocab: usize,
    n_audio_ctx: usize,
    n_audio_state: usize,
    n_audio_head: usize,
    n_audio_layer: usize,
    n_text_ctx: usize,
    n_text_state: usize,
    n_text_head: usize,
    n_text_layer: usize,
    n_mels: usize,
    ftype: usize,

    /// Largest value any hyperparameter may have. Real models are far below
    /// it (vocab 51866, state 1280, contexts 1500); keeping every field this
    /// small means no size computed from them can overflow (the library is
    /// built without runtime safety checks).
    pub const max_value = 1 << 17;

    /// Rejects hyperparameters a model file can't sensibly have: zero head
    /// counts (later divisors), widths not divisible by their head count,
    /// and anything over `max_value`.
    pub fn check(hp: HParams) !void {
        inline for (@typeInfo(HParams).@"struct".field_names) |name| {
            if (@field(hp, name) > max_value) return error.BadValue;
        }
        if (hp.n_audio_head == 0 or hp.n_text_head == 0) return error.BadValue;
        if (hp.n_audio_state % hp.n_audio_head != 0 or hp.n_text_state % hp.n_text_head != 0) return error.BadValue;
    }
};

pub const Norm = struct { w: []const f32, b: []const f32 };
pub const Attn = struct { q: Linear, k: Linear, v: Linear, o: Linear };

pub const EncoderLayer = struct {
    attn_ln: Norm,
    attn: Attn,
    mlp_ln: Norm,
    fc1: Linear,
    fc2: Linear,
};

pub const DecoderLayer = struct {
    attn_ln: Norm,
    attn: Attn,
    cross_ln: Norm,
    cross: Attn,
    mlp_ln: Norm,
    fc1: Linear,
    fc2: Linear,
};

/// Special token ids. They shift depending on whether the model is
/// English-only or multilingual (and on the number of languages for v3).
pub const Tokens = struct {
    eot: u32,
    sot: u32,
    translate: u32,
    transcribe: u32,
    solm: u32,
    prev: u32,
    nosp: u32,
    not: u32,
    beg: u32,
    lang_en: u32,
    multilingual: bool,

    fn init(n_vocab: usize) Tokens {
        const multi = n_vocab >= 51865;
        const m: u32 = if (multi) 1 else 0;
        // large-v3 has 100 languages instead of 99.
        const dt: u32 = if (multi) @intCast(n_vocab - 51865) else 0;
        return .{
            .eot = 50256 + m,
            .sot = 50257 + m,
            .lang_en = 50258 + m,
            .translate = 50357 + m + dt,
            .transcribe = 50358 + m + dt,
            .solm = 50359 + m + dt,
            .prev = 50360 + m + dt,
            .nosp = 50361 + m + dt,
            .not = 50362 + m + dt,
            .beg = 50363 + m + dt,
            .multilingual = multi,
        };
    }
};

pub const Model = struct {
    /// Heap-allocated so the arena never moves (the struct is returned by value).
    arena: *std.heap.ArenaAllocator,
    hp: HParams,
    tokens: Tokens,
    /// [n_mels][n_fft_bins] mel filterbank (n_fft_bins = 201).
    mel_filters: []const f32,
    n_fft_bins: usize,
    /// Byte strings for each regular token id.
    vocab: []const []const u8,

    // Encoder
    conv1: Linear, // as im2col matmul: [n_state][n_mels*3]
    conv2: Linear, // [n_state][n_state*3]
    enc_pos: []const f32, // [n_audio_ctx][n_state]
    enc_layers: []EncoderLayer,
    enc_ln_post: Norm,

    // Decoder
    tok_emb: []const f16, // [n_vocab][n_state]
    dec_pos: []const f32, // [n_text_ctx][n_state]
    dec_layers: []DecoderLayer,
    dec_ln: Norm,

    pub fn deinit(self: *Model) void {
        const gpa = self.arena.child_allocator;
        self.arena.deinit();
        gpa.destroy(self.arena);
        self.* = undefined;
    }

    /// Loads a ggml model file (f16 or f32; quantized files unsupported).
    pub fn load(gpa: Allocator, io: std.Io, path: []const u8) !Model {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(8 << 30));
        defer gpa.free(bytes);
        return parse(gpa, bytes);
    }

    pub fn parse(gpa: Allocator, bytes: []const u8) !Model {
        const arena = try gpa.create(std.heap.ArenaAllocator);
        errdefer gpa.destroy(arena);
        arena.* = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        var r: Cursor = .{ .buf = bytes };

        if (try r.int(u32) != 0x67676d6c) return error.BadMagic;
        var hp: HParams = undefined;
        inline for (@typeInfo(HParams).@"struct".field_names) |name| {
            @field(hp, name) = try r.usz();
        }
        if (hp.ftype > 1) return error.QuantizedModelUnsupported;
        try hp.check();

        const n_mel = try r.usz();
        const n_fft = try r.usz();
        const filters = try a.alloc(f32, std.math.mul(usize, n_mel, n_fft) catch return error.BadValue);
        for (filters) |*f| f.* = @bitCast(try r.int(u32));

        const n_vocab_file = try r.usz();
        const vocab = try a.alloc([]const u8, n_vocab_file);
        for (vocab) |*v| {
            const len = try r.int(u32);
            v.* = try a.dupe(u8, try r.take(len));
        }

        var tensors: std.StringHashMapUnmanaged(Raw) = .empty;
        defer tensors.deinit(gpa);
        while (r.pos < bytes.len) {
            const n_dims = try r.usz();
            const name_len = try r.usz();
            const ttype = try r.usz();
            if (n_dims > 4) return error.BadTensor;
            var count: usize = 1;
            for (0..n_dims) |_| count = std.math.mul(usize, count, try r.usz()) catch return error.BadTensor;
            const name = try r.take(name_len);
            const raw: Raw = switch (ttype) {
                0 => .{ .f32 = try r.take(std.math.mul(usize, count, 4) catch return error.BadTensor) },
                1 => .{ .f16 = try r.take(std.math.mul(usize, count, 2) catch return error.BadTensor) },
                else => return error.UnsupportedTensorType,
            };
            try tensors.put(gpa, name, raw);
        }

        var b: Builder = .{ .t = &tensors, .a = a };
        const ns = hp.n_audio_state;
        const ts = hp.n_text_state;

        const enc_layers = try a.alloc(EncoderLayer, hp.n_audio_layer);
        for (enc_layers, 0..) |*l, i| {
            const p = "encoder.blocks.{d}.";
            l.* = .{
                .attn_ln = try b.norm(p ++ "attn_ln", i, ns),
                .attn = try b.attn(p ++ "attn", i, ns),
                .mlp_ln = try b.norm(p ++ "mlp_ln", i, ns),
                .fc1 = try b.linear(p ++ "mlp.0", i, ns, 4 * ns, true),
                .fc2 = try b.linear(p ++ "mlp.2", i, 4 * ns, ns, true),
            };
        }
        const dec_layers = try a.alloc(DecoderLayer, hp.n_text_layer);
        for (dec_layers, 0..) |*l, i| {
            const p = "decoder.blocks.{d}.";
            l.* = .{
                .attn_ln = try b.norm(p ++ "attn_ln", i, ts),
                .attn = try b.attn(p ++ "attn", i, ts),
                .cross_ln = try b.norm(p ++ "cross_attn_ln", i, ts),
                .cross = try b.attn(p ++ "cross_attn", i, ts),
                .mlp_ln = try b.norm(p ++ "mlp_ln", i, ts),
                .fc1 = try b.linear(p ++ "mlp.0", i, ts, 4 * ts, true),
                .fc2 = try b.linear(p ++ "mlp.2", i, 4 * ts, ts, true),
            };
        }

        return .{
            .arena = arena,
            .hp = hp,
            .tokens = .init(hp.n_vocab),
            .mel_filters = filters,
            .n_fft_bins = n_fft,
            .vocab = vocab,
            // Conv1d weights [out][in][3] flatten to [out][in*3], which is
            // exactly a Linear layer over im2col'd input.
            .conv1 = .{ .w = try b.mat("encoder.conv1.weight", ns * hp.n_mels * 3), .b = try b.get("encoder.conv1.bias", ns), .n_in = hp.n_mels * 3, .n_out = ns },
            .conv2 = .{ .w = try b.mat("encoder.conv2.weight", ns * ns * 3), .b = try b.get("encoder.conv2.bias", ns), .n_in = ns * 3, .n_out = ns },
            .enc_pos = try b.get("encoder.positional_embedding", hp.n_audio_ctx * ns),
            .enc_layers = enc_layers,
            .enc_ln_post = .{ .w = try b.get("encoder.ln_post.weight", ns), .b = try b.get("encoder.ln_post.bias", ns) },
            .tok_emb = try b.mat("decoder.token_embedding.weight", hp.n_vocab * ts),
            .dec_pos = try b.get("decoder.positional_embedding", hp.n_text_ctx * ts),
            .dec_layers = dec_layers,
            .dec_ln = .{ .w = try b.get("decoder.ln.weight", ts), .b = try b.get("decoder.ln.bias", ts) },
        };
    }
};

test "model header: zero or oversized hyperparameters are rejected" {
    const gpa = std.testing.allocator;
    var buf: [4 + 11 * 4]u8 = undefined;
    std.mem.writeInt(u32, buf[0..4], 0x67676d6c, .little);
    const good = [11]i32{ 51864, 1500, 384, 6, 4, 448, 384, 6, 4, 80, 1 };
    const cases = [_]struct { field: usize, value: i32 }{
        .{ .field = 3, .value = 0 }, // n_audio_head = 0 (a divisor)
        .{ .field = 7, .value = 0 }, // n_text_head = 0
        .{ .field = 7, .value = 7 }, // 384 % 7 != 0
        .{ .field = 0, .value = std.math.maxInt(i32) }, // n_vocab: sizes would overflow
    };
    for (cases) |cs| {
        var hp = good;
        hp[cs.field] = cs.value;
        for (hp, 0..) |v, i| std.mem.writeInt(i32, buf[4 + i * 4 ..][0..4], v, .little);
        try std.testing.expectError(error.BadValue, Model.parse(gpa, &buf));
    }
    // The good header passes the check (and fails later: no filterbank).
    for (good, 0..) |v, i| std.mem.writeInt(i32, buf[4 + i * 4 ..][0..4], v, .little);
    try std.testing.expectError(error.UnexpectedEof, Model.parse(gpa, &buf));
}

const Cursor = struct {
    buf: []const u8,
    pos: usize = 0,

    fn take(self: *Cursor, n: usize) ![]const u8 {
        if (self.buf.len - self.pos < n) return error.UnexpectedEof;
        defer self.pos += n;
        return self.buf[self.pos..][0..n];
    }
    fn int(self: *Cursor, comptime T: type) !T {
        return std.mem.readInt(T, (try self.take(@sizeOf(T)))[0..@sizeOf(T)], .little);
    }
    fn usz(self: *Cursor) !usize {
        const v = try self.int(i32);
        if (v < 0) return error.BadValue;
        return @intCast(v);
    }
};

/// A tensor's bytes in the file (little-endian).
const Raw = union(enum) {
    f16: []const u8,
    f32: []const u8,

    fn len(raw: Raw) usize {
        return switch (raw) {
            .f16 => |b| b.len / 2,
            .f32 => |b| b.len / 4,
        };
    }
};

const Builder = struct {
    t: *std.StringHashMapUnmanaged(Raw),
    a: Allocator,
    buf: [128]u8 = undefined,

    fn raw(self: *Builder, name: []const u8, expect_len: usize) !Raw {
        const data = self.t.get(name) orelse {
            std.log.err("model is missing tensor '{s}'", .{name});
            return error.MissingTensor;
        };
        if (data.len() != expect_len) {
            std.log.err("tensor '{s}' has {d} elements, expected {d}", .{ name, data.len(), expect_len });
            return error.BadTensorShape;
        }
        return data;
    }
    /// A vector (bias, norm, positional table) as f32.
    fn get(self: *Builder, name: []const u8, expect_len: usize) ![]const f32 {
        const out = try self.a.alloc(f32, expect_len);
        switch (try self.raw(name, expect_len)) {
            .f32 => |b| for (out, 0..) |*d, i| {
                d.* = @bitCast(std.mem.readInt(u32, b[i * 4 ..][0..4], .little));
            },
            .f16 => |b| for (out, 0..) |*d, i| {
                d.* = @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, b[i * 2 ..][0..2], .little))));
            },
        }
        return out;
    }
    /// A weight matrix as f16 (f32 files are rounded; the GPU computes with f16 weights).
    fn mat(self: *Builder, name: []const u8, expect_len: usize) ![]const f16 {
        const out = try self.a.alloc(f16, expect_len);
        switch (try self.raw(name, expect_len)) {
            .f16 => |b| for (out, 0..) |*d, i| {
                d.* = @bitCast(std.mem.readInt(u16, b[i * 2 ..][0..2], .little));
            },
            .f32 => |b| for (out, 0..) |*d, i| {
                d.* = @floatCast(@as(f32, @bitCast(std.mem.readInt(u32, b[i * 4 ..][0..4], .little))));
            },
        }
        return out;
    }
    fn tname(self: *Builder, comptime fmt: []const u8, i: usize, suffix: []const u8) ![]const u8 {
        const base = try std.fmt.bufPrint(&self.buf, fmt, .{i});
        const len = base.len;
        @memcpy(self.buf[len..][0..suffix.len], suffix);
        return self.buf[0 .. len + suffix.len];
    }
    fn norm(self: *Builder, comptime fmt: []const u8, i: usize, n: usize) !Norm {
        return .{ .w = try self.get(try self.tname(fmt, i, ".weight"), n), .b = try self.get(try self.tname(fmt, i, ".bias"), n) };
    }
    fn linear(self: *Builder, comptime fmt: []const u8, i: usize, n_in: usize, n_out: usize, bias: bool) !Linear {
        return .{
            .w = try self.mat(try self.tname(fmt, i, ".weight"), n_in * n_out),
            .b = if (bias) try self.get(try self.tname(fmt, i, ".bias"), n_out) else null,
            .n_in = n_in,
            .n_out = n_out,
        };
    }
    fn attn(self: *Builder, comptime fmt: []const u8, i: usize, n: usize) !Attn {
        return .{
            .q = try self.linear(fmt ++ ".query", i, n, n, true),
            .k = try self.linear(fmt ++ ".key", i, n, n, false), // key projection has no bias
            .v = try self.linear(fmt ++ ".value", i, n, n, true),
            .o = try self.linear(fmt ++ ".out", i, n, n, true),
        };
    }
};
