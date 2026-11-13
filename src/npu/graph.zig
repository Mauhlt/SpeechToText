//! A small op-by-op graph builder over QNN's built-in ops (package
//! qti.aisw). Every tensor of the network uses one floating-point type:
//! f16 on the HTP (what the NPU computes in) and f32 on QNN's CPU reference
//! backend (which has no f16 kernels). Static data (weights) is converted to
//! that type on the host while the graph is built.
//!
//! Shapes are tracked here, so each op helper returns its output tensor.
//! Graph outputs are made with `output`, which copies a tensor into an
//! APP_READ tensor through a Reshape (the backend elides it).
const std = @import("std");
const qnn = @import("qnn");
const c = qnn.c;
const ops = qnn.ops;
const Allocator = std.mem.Allocator;

pub const DType = enum {
    f16,
    f32,

    pub fn qnnType(d: DType) c.Qnn_DataType_t {
        return switch (d) {
            .f16 => c.QNN_DATATYPE_FLOAT_16,
            .f32 => c.QNN_DATATYPE_FLOAT_32,
        };
    }
    pub fn size(d: DType) usize {
        return switch (d) {
            .f16 => 2,
            .f32 => 4,
        };
    }
};

/// A tensor of the graph being built.
pub const T = struct {
    raw: c.Qnn_Tensor_t,
    dims: []const u32,

    pub fn rank(t: T) usize {
        return t.dims.len;
    }
    pub fn count(t: T) usize {
        var n: usize = 1;
        for (t.dims) |d| n *= d;
        return n;
    }
};

pub const Error = error{ QnnError, OutOfMemory, BadShape };

/// The name of the op being added when the last error happened (for messages).
pub threadlocal var failed_op: [64]u8 = @splat(0);

pub const Builder = struct {
    api: *const qnn.Interface,
    graph: c.Qnn_GraphHandle_t,
    /// Owns names, dims and static data; must live until graphFinalize returns.
    arena: Allocator,
    dt: DType,
    n_tensors: u32 = 0,
    n_nodes: u32 = 0,
    /// APP_WRITE / APP_READ tensors in creation order.
    inputs: std.ArrayList(c.Qnn_Tensor_t) = .empty,
    outputs: std.ArrayList(c.Qnn_Tensor_t) = .empty,

    fn create(b: *Builder, name: ?[:0]const u8, kind: qnn.TensorKind, data_type: c.Qnn_DataType_t, dims_in: []const u32, data: ?[]u8) Error!T {
        const dims = try b.arena.dupe(u32, dims_in);
        const nm: [:0]const u8 = name orelse try std.fmt.allocPrintSentinel(b.arena, "t{d}", .{b.n_tensors}, 0);
        b.n_tensors += 1;
        var t = qnn.tensor(nm.ptr, kind, data_type, dims, data);
        const st = b.api.tensorCreateGraphTensor.?(b.graph, &t);
        if (st & 0xffff != 0) {
            setFailed(nm);
            std.log.err("QNN: creating tensor {s} {any} failed (status {d})", .{ nm, dims, st & 0xffff });
            qnn.last_error = st & 0xffff;
            return error.QnnError;
        }
        switch (kind) {
            .app_write => try b.inputs.append(b.arena, t),
            .app_read => try b.outputs.append(b.arena, t),
            else => {},
        }
        return .{ .raw = t, .dims = dims };
    }

    /// A graph input (APP_WRITE) in the network's type.
    pub fn input(b: *Builder, name: [:0]const u8, dims: []const u32) Error!T {
        return b.create(name, .app_write, b.dt.qnnType(), dims, null);
    }

    /// A graph output: `x` copied into an APP_READ tensor of shape `dims`
    /// (same element count).
    pub fn output(b: *Builder, name: [:0]const u8, x: T, dims: []const u32) Error!T {
        const out = try b.create(name, .app_read, b.dt.qnnType(), dims, null);
        if (countOf(dims) != x.count()) return error.BadShape;
        try b.addNode(ops.reshape.name, &.{}, &.{x}, out);
        return out;
    }

    /// A graph input of an explicit type (e.g. INT_32 indices).
    pub fn inputTyped(b: *Builder, name: [:0]const u8, data_type: c.Qnn_DataType_t, dims: []const u32) Error!T {
        return b.create(name, .app_write, data_type, dims, null);
    }

    /// Adds node `op` whose output is directly a graph output (APP_READ).
    pub fn nodeOut(b: *Builder, op: [:0]const u8, params: []const c.Qnn_Param_t, ins: []const T, name: [:0]const u8, data_type: c.Qnn_DataType_t, dims: []const u32) Error!T {
        const out = try b.create(name, .app_read, data_type, dims, null);
        try b.addNode(op, params, ins, out);
        return out;
    }

    /// Rows of `data` [n][d] picked by `idx` [k] (INT_32) -> [k][d].
    pub fn gather(b: *Builder, data: T, idx: T) Error!T {
        if (data.rank() != 2 or idx.rank() != 1) return error.BadShape;
        return b.node(ops.gather.name, &.{qnn.scalarParam(ops.gather.param.axis, i32, 0)}, &.{ data, idx }, &.{ idx.dims[0], data.dims[1] });
    }

    fn native(b: *Builder, dims: []const u32) Error!T {
        return b.create(null, .native, b.dt.qnnType(), dims, null);
    }

    /// Static f32 data, stored in the network's type.
    pub fn constF32(b: *Builder, dims: []const u32, data: []const f32) Error!T {
        if (countOf(dims) != data.len) return error.BadShape;
        return b.create(null, .static, b.dt.qnnType(), dims, try b.convert(f32, data));
    }

    /// Static f16 data, stored in the network's type. With f16 the caller's
    /// buffer is referenced, not copied: it must live until finalize.
    pub fn constF16(b: *Builder, dims: []const u32, data: []const f16) Error!T {
        if (countOf(dims) != data.len) return error.BadShape;
        const bytes: []u8 = if (b.dt == .f16) @constCast(std.mem.sliceAsBytes(data)) else try b.convert(f16, data);
        return b.create(null, .static, b.dt.qnnType(), dims, bytes);
    }

    /// A one-element constant, for broadcasting.
    pub fn scalar(b: *Builder, v: f32) Error!T {
        return b.constF32(&.{1}, &.{v});
    }

    fn convert(b: *Builder, comptime S: type, data: []const S) Error![]u8 {
        switch (b.dt) {
            .f16 => {
                const out = try b.arena.alloc(f16, data.len);
                for (out, data) |*o, v| o.* = @floatCast(v);
                return std.mem.sliceAsBytes(out);
            },
            .f32 => {
                const out = try b.arena.alloc(f32, data.len);
                for (out, data) |*o, v| o.* = @floatCast(v);
                return std.mem.sliceAsBytes(out);
            },
        }
    }

    /// A static u32 / i32 tensor for a tensor parameter.
    fn paramTensor(b: *Builder, comptime I: type, dims: []const u32, data: []const I) Error!c.Qnn_Tensor_t {
        const dt: c.Qnn_DataType_t = if (I == u32) c.QNN_DATATYPE_UINT_32 else c.QNN_DATATYPE_INT_32;
        const t = try b.create(null, .static, dt, dims, std.mem.sliceAsBytes(try b.arena.dupe(I, data)));
        return t.raw;
    }

    fn addNode(b: *Builder, op: [:0]const u8, params: []const c.Qnn_Param_t, ins: []const T, out: T) Error!void {
        const name = try std.fmt.allocPrintSentinel(b.arena, "n{d}_{s}", .{ b.n_nodes, op }, 0);
        b.n_nodes += 1;
        const p = try b.arena.dupe(c.Qnn_Param_t, params);
        const in = try b.arena.alloc(c.Qnn_Tensor_t, ins.len);
        for (in, ins) |*d, s| d.* = s.raw;
        const o = try b.arena.alloc(c.Qnn_Tensor_t, 1);
        o[0] = out.raw;
        const st = b.api.graphAddNode.?(b.graph, qnn.opConfig(name.ptr, op.ptr, p, in, o));
        if (st & 0xffff != 0) {
            setFailed(name);
            var shapes: [8][]const u32 = undefined;
            for (ins[0..@min(ins.len, 8)], 0..) |t, i| shapes[i] = t.dims;
            std.log.err("QNN: adding node {s} failed (status {d}); inputs {any} -> {any}", .{ name, st & 0xffff, shapes[0..@min(ins.len, 8)], out.dims });
            qnn.last_error = st & 0xffff;
            return error.QnnError;
        }
    }

    /// Adds node `op` with a fresh native output of shape `dims`.
    pub fn node(b: *Builder, op: [:0]const u8, params: []const c.Qnn_Param_t, ins: []const T, dims: []const u32) Error!T {
        const out = try b.native(dims);
        try b.addNode(op, params, ins, out);
        return out;
    }

    // ------------------------------------------------------------ ops

    /// y[.., n_out] = x[.., n_in] · wᵀ + bias, w stored [n_out][n_in].
    pub fn fc(b: *Builder, x: T, w: T, bias: ?T) Error!T {
        if (w.rank() != 2 or x.dims[x.rank() - 1] != w.dims[1]) return error.BadShape;
        const rows = x.count() / w.dims[1];
        const y = if (bias) |bb|
            try b.node(ops.fully_connected.name, &.{}, &.{ x, w, bb }, &.{ @intCast(rows), w.dims[0] })
        else
            try b.node(ops.fully_connected.name, &.{}, &.{ x, w }, &.{ @intCast(rows), w.dims[0] });
        if (x.rank() == 2) return y;
        var dims = try b.arena.dupe(u32, x.dims);
        dims[dims.len - 1] = w.dims[0];
        return b.reshape(y, dims);
    }

    fn broadcast(b: *Builder, x: T, y: T) Error![]u32 {
        const r = @max(x.rank(), y.rank());
        const dims = try b.arena.alloc(u32, r);
        for (0..r) |i| {
            const dx = if (i + x.rank() >= r) x.dims[i + x.rank() - r] else 1;
            const dy = if (i + y.rank() >= r) y.dims[i + y.rank() - r] else 1;
            if (dx != dy and dx != 1 and dy != 1) return error.BadShape;
            dims[i] = @max(dx, dy);
        }
        return dims;
    }

    fn binary(b: *Builder, op: [:0]const u8, x: T, y: T) Error!T {
        return b.node(op, &.{}, &.{ x, y }, try b.broadcast(x, y));
    }
    pub fn add(b: *Builder, x: T, y: T) Error!T {
        return b.binary(ops.element_wise_add.name, x, y);
    }
    pub fn sub(b: *Builder, x: T, y: T) Error!T {
        return b.binary(ops.element_wise_subtract.name, x, y);
    }
    pub fn mul(b: *Builder, x: T, y: T) Error!T {
        return b.binary(ops.element_wise_multiply.name, x, y);
    }
    pub fn max(b: *Builder, x: T, y: T) Error!T {
        return b.binary(ops.element_wise_maximum.name, x, y);
    }
    pub fn unary(b: *Builder, op: [:0]const u8, x: T) Error!T {
        return b.node(op, &.{}, &.{x}, x.dims);
    }

    /// LayerNorm over the last axis, epsilon 1e-5.
    pub fn layerNorm(b: *Builder, x: T, gamma: T, beta: T) Error!T {
        const axes = try b.paramTensor(u32, &.{1}, &.{@intCast(x.rank() - 1)});
        return b.node(ops.layer_norm.name, &.{
            qnn.scalarParam(ops.layer_norm.param.epsilon, f32, 1e-5),
            qnn.tensorParam(ops.layer_norm.param.axes, axes),
        }, &.{ x, gamma, beta }, x.dims);
    }

    /// Batched matrix product over the last two axes.
    pub fn matmul(b: *Builder, x: T, y: T, tr0: bool, tr1: bool) Error!T {
        const r = x.rank();
        if (y.rank() != r or r < 2) return error.BadShape;
        const m = if (tr0) x.dims[r - 1] else x.dims[r - 2];
        const k0 = if (tr0) x.dims[r - 2] else x.dims[r - 1];
        const k1 = if (tr1) y.dims[r - 1] else y.dims[r - 2];
        const n = if (tr1) y.dims[r - 2] else y.dims[r - 1];
        if (k0 != k1) return error.BadShape;
        const dims = try b.arena.dupe(u32, x.dims);
        dims[r - 2] = m;
        dims[r - 1] = n;
        return b.node(ops.mat_mul.name, &.{
            qnn.scalarParam(ops.mat_mul.param.transpose_in0, bool, tr0),
            qnn.scalarParam(ops.mat_mul.param.transpose_in1, bool, tr1),
        }, &.{ x, y }, dims);
    }

    /// Softmax over the last axis.
    pub fn softmax(b: *Builder, x: T) Error!T {
        return b.node(ops.softmax.name, &.{
            qnn.scalarParam(ops.softmax.param.axis, u32, @intCast(x.rank() - 1)),
            qnn.scalarParam(ops.softmax.param.beta, f32, 1.0),
        }, &.{x}, x.dims);
    }

    pub fn reshape(b: *Builder, x: T, dims: []const u32) Error!T {
        if (countOf(dims) != x.count()) return error.BadShape;
        return b.node(ops.reshape.name, &.{}, &.{x}, dims);
    }

    pub fn transpose(b: *Builder, x: T, perm: []const u32) Error!T {
        if (perm.len != x.rank()) return error.BadShape;
        const dims = try b.arena.alloc(u32, perm.len);
        for (dims, perm) |*d, p| d.* = x.dims[p];
        const pt = try b.paramTensor(u32, &.{@intCast(perm.len)}, perm);
        return b.node(ops.transpose.name, &.{qnn.tensorParam(ops.transpose.param.perm, pt)}, &.{x}, dims);
    }

    pub fn concat(b: *Builder, xs: []const T, axis: u32) Error!T {
        const dims = try b.arena.dupe(u32, xs[0].dims);
        dims[axis] = 0;
        for (xs) |x| dims[axis] += x.dims[axis];
        return b.node(ops.concat.name, &.{qnn.scalarParam(ops.concat.param.axis, u32, axis)}, xs, dims);
    }

    /// x[begin[i]..end[i]] along every axis (stride 1).
    pub fn slice(b: *Builder, x: T, begin: []const u32, end: []const u32) Error!T {
        const r = x.rank();
        if (begin.len != r or end.len != r) return error.BadShape;
        const ranges = try b.arena.alloc(i32, r * 3);
        const dims = try b.arena.alloc(u32, r);
        for (0..r) |i| {
            if (end[i] > x.dims[i] or begin[i] >= end[i]) return error.BadShape;
            ranges[i * 3 + 0] = @intCast(begin[i]);
            ranges[i * 3 + 1] = @intCast(end[i]);
            ranges[i * 3 + 2] = 1;
            dims[i] = end[i] - begin[i];
        }
        const rt = try b.paramTensor(i32, &.{ @intCast(r), 3 }, ranges);
        return b.node(ops.strided_slice.name, &.{qnn.tensorParam(ops.strided_slice.param.ranges, rt)}, &.{x}, dims);
    }

    /// Max over all elements, as a [1, 1, ...] tensor of the same rank.
    pub fn reduceMaxAll(b: *Builder, x: T) Error!T {
        const axes = try b.arena.alloc(u32, x.rank());
        const dims = try b.arena.alloc(u32, x.rank());
        for (axes, dims, 0..) |*a, *d, i| {
            a.* = @intCast(i);
            d.* = 1;
        }
        const at = try b.paramTensor(u32, &.{@intCast(axes.len)}, axes);
        return b.node(ops.reduce_max.name, &.{
            qnn.tensorParam(ops.reduce_max.param.axes, at),
            qnn.scalarParam(ops.reduce_max.param.keep_dims, bool, true),
        }, &.{x}, dims);
    }

    /// 2-D convolution, NHWC input [1][h][w][cin], filter [fh][fw][cin][cout].
    pub fn conv2d(b: *Builder, x: T, w: T, bias: T, stride: [2]u32, pad: [2][2]u32) Error!T {
        if (x.rank() != 4 or w.rank() != 4 or x.dims[3] != w.dims[2]) return error.BadShape;
        const oh = (x.dims[1] + pad[0][0] + pad[0][1] - w.dims[0]) / stride[0] + 1;
        const ow = (x.dims[2] + pad[1][0] + pad[1][1] - w.dims[1]) / stride[1] + 1;
        const st = try b.paramTensor(u32, &.{2}, &stride);
        const pd = try b.paramTensor(u32, &.{ 2, 2 }, &.{ pad[0][0], pad[0][1], pad[1][0], pad[1][1] });
        const dl = try b.paramTensor(u32, &.{2}, &.{ 1, 1 });
        return b.node(ops.conv_2d.name, &.{
            qnn.tensorParam(ops.conv_2d.param.stride, st),
            qnn.tensorParam(ops.conv_2d.param.pad_amount, pd),
            qnn.tensorParam(ops.conv_2d.param.dilation, dl),
            qnn.scalarParam(ops.conv_2d.param.group, u32, 1),
        }, &.{ x, w, bias }, &.{ x.dims[0], oh, ow, w.dims[3] });
    }
};

fn countOf(dims: []const u32) usize {
    var n: usize = 1;
    for (dims) |d| n *= d;
    return n;
}

fn setFailed(name: []const u8) void {
    const n = @min(name.len, failed_op.len - 1);
    @memcpy(failed_op[0..n], name[0..n]);
    failed_op[n] = 0;
}
