//! The personal LoRA adapter: low-rank matrices on every decoder linear
//! layer, W' = W + (alpha/rank)·B·A. This file holds the host-side data and
//! its file format (`adapter.bin`); training (gpu/trainer.zig) and merging into
//! the weights (gpu/engine.zig) happen on the GPU.
const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const model_mod = @import("model.zig");
const Model = model_mod.Model;
const Linear = model_mod.Linear;

/// Decoder layers that get an adapter, in file order.
pub const Slot = enum { q, k, v, o, cross_q, cross_o, fc1, fc2 };
pub const n_slots = @typeInfo(Slot).@"enum".field_names.len;

pub fn slotLinear(m: *const Model, layer: usize, slot: Slot) Linear {
    const l = m.dec_layers[layer];
    return switch (slot) {
        .q => l.attn.q,
        .k => l.attn.k,
        .v => l.attn.v,
        .o => l.attn.o,
        .cross_q => l.cross.q,
        .cross_o => l.cross.o,
        .fc1 => l.fc1,
        .fc2 => l.fc2,
    };
}

pub const Lora = struct {
    a: []f32, // [rank][n_in]
    b: []f32, // [n_out][rank]
    n_in: usize,
    n_out: usize,
};

pub const Adapter = struct {
    rank: usize,
    alpha: f32,
    n_layers: usize,
    n_state: usize,
    loras: []Lora, // [n_layers * n_slots]
    /// All parameters, contiguous: each Lora's `a` then `b`, in slot order.
    buf: []f32,

    pub const default_rank = 8;
    pub const default_alpha = 16;

    pub fn scale(ad: *const Adapter) f32 {
        return ad.alpha / @as(f32, @floatFromInt(ad.rank));
    }

    pub fn get(ad: *const Adapter, layer: usize, slot: Slot) *Lora {
        return &ad.loras[layer * n_slots + @intFromEnum(slot)];
    }

    /// Offset of a slot's `a` (and `b` right after it) in `buf`, in floats.
    pub fn offsetOf(ad: *const Adapter, layer: usize, slot: Slot) usize {
        return (@intFromPtr(ad.get(layer, slot).a.ptr) - @intFromPtr(ad.buf.ptr)) / 4;
    }

    /// New adapter: A random, B zero (so it starts as an exact no-op).
    pub fn init(gpa: Allocator, m: *const Model, rank: usize, alpha: f32, seed: u64) !Adapter {
        const ad = try alloc(gpa, m, rank, alpha);
        var rng: std.Random.DefaultPrng = .init(seed);
        const r = rng.random();
        for (ad.loras) |lo| {
            const bound = 1.0 / @sqrt(@as(f32, @floatFromInt(lo.n_in)));
            for (lo.a) |*x| x.* = (r.float(f32) * 2 - 1) * bound;
            @memset(lo.b, 0);
        }
        return ad;
    }

    /// Number of f32 parameters in an adapter of this rank for model `m`.
    pub fn paramCount(m: *const Model, rank: usize) usize {
        var total: usize = 0;
        for (0..m.hp.n_text_layer) |l| for (0..n_slots) |si| {
            const lin = slotLinear(m, l, @enumFromInt(si));
            total += rank * (lin.n_in + lin.n_out);
        };
        return total;
    }

    fn alloc(gpa: Allocator, m: *const Model, rank: usize, alpha: f32) !Adapter {
        const nl = m.hp.n_text_layer;
        const loras = try gpa.alloc(Lora, nl * n_slots);
        errdefer gpa.free(loras);
        const buf = try gpa.alloc(f32, paramCount(m, rank));
        var off: usize = 0;
        for (0..nl) |l| for (0..n_slots) |si| {
            const lin = slotLinear(m, l, @enumFromInt(si));
            const lo = &loras[l * n_slots + si];
            lo.* = .{ .a = buf[off..][0 .. rank * lin.n_in], .b = buf[off + rank * lin.n_in ..][0 .. rank * lin.n_out], .n_in = lin.n_in, .n_out = lin.n_out };
            off += rank * (lin.n_in + lin.n_out);
        };
        return .{ .rank = rank, .alpha = alpha, .n_layers = nl, .n_state = m.hp.n_text_state, .loras = loras, .buf = buf };
    }

    pub fn deinit(ad: *Adapter, gpa: Allocator) void {
        gpa.free(ad.loras);
        gpa.free(ad.buf);
        ad.* = undefined;
    }

    const magic = "STTLORA1";

    /// Writes the adapter via a temp file + rename, so a crash mid-save can't
    /// leave a truncated adapter behind.
    pub fn save(ad: *const Adapter, gpa: Allocator, io: Io, path: []const u8) !void {
        const bytes = try ad.serialize(gpa);
        defer gpa.free(bytes);
        const tmp = try std.mem.concat(gpa, u8, &.{ path, ".tmp" });
        defer gpa.free(tmp);
        const cwd = Io.Dir.cwd();
        try cwd.writeFile(io, .{ .sub_path = tmp, .data = bytes });
        try cwd.rename(tmp, cwd, path, io);
    }

    pub fn serialize(ad: *const Adapter, gpa: Allocator) ![]u8 {
        const bytes = try gpa.alloc(u8, 8 + 16 + ad.buf.len * 4);
        @memcpy(bytes[0..8], magic);
        std.mem.writeInt(u32, bytes[8..12], @intCast(ad.n_layers), .little);
        std.mem.writeInt(u32, bytes[12..16], @intCast(ad.n_state), .little);
        std.mem.writeInt(u32, bytes[16..20], @intCast(ad.rank), .little);
        std.mem.writeInt(u32, bytes[20..24], @bitCast(ad.alpha), .little);
        for (ad.buf, 0..) |v, i| std.mem.writeInt(u32, bytes[24 + i * 4 ..][0..4], @bitCast(v), .little);
        return bytes;
    }

    pub fn load(gpa: Allocator, io: Io, path: []const u8, m: *const Model) !Adapter {
        const bytes = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
        defer gpa.free(bytes);
        return parse(gpa, bytes, m);
    }

    /// Checks an adapter file without the model: magic, rank, alpha, and
    /// that its size matches its own header (every decoder layer holds six
    /// n_state x n_state slots and fc1 / fc2 of n_state x 4·n_state, so
    /// 22·n_state parameters per layer per rank).
    pub fn check(bytes: []const u8) !void {
        if (bytes.len < 24 or !std.mem.eql(u8, bytes[0..8], magic)) return error.BadAdapterFile;
        const nl: usize = std.mem.readInt(u32, bytes[8..12], .little);
        const ns: usize = std.mem.readInt(u32, bytes[12..16], .little);
        const rank: usize = std.mem.readInt(u32, bytes[16..20], .little);
        const alpha: f32 = @bitCast(std.mem.readInt(u32, bytes[20..24], .little));
        if (nl == 0 or ns == 0 or rank == 0 or rank > max_rank or !std.math.isFinite(alpha)) return error.BadAdapterFile;
        const params = std.math.mul(usize, 22 * rank, std.math.mul(usize, nl, ns) catch return error.BadAdapterFile) catch return error.BadAdapterFile;
        const size = std.math.mul(usize, params, 4) catch return error.BadAdapterFile;
        if (bytes.len - 24 != size) return error.BadAdapterFile;
    }

    pub fn parse(gpa: Allocator, bytes: []const u8, m: *const Model) !Adapter {
        // Validate the header against the file size before allocating anything.
        try check(bytes);
        const nl = std.mem.readInt(u32, bytes[8..12], .little);
        const ns = std.mem.readInt(u32, bytes[12..16], .little);
        if (nl != m.hp.n_text_layer or ns != m.hp.n_text_state) return error.AdapterModelMismatch;
        const rank = std.mem.readInt(u32, bytes[16..20], .little);
        const alpha: f32 = @bitCast(std.mem.readInt(u32, bytes[20..24], .little));
        if (bytes.len != 24 + paramCount(m, rank) * 4) return error.BadAdapterFile;
        var ad = try alloc(gpa, m, rank, alpha);
        errdefer ad.deinit(gpa);
        for (ad.buf, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, bytes[24 + i * 4 ..][0..4], .little));
        return ad;
    }

    /// The GPU optimizer handles ranks up to this.
    pub const max_rank = 32;
};
