//! Minimal binding to the CUDA *driver* API (nvcuda.dll / libcuda.so), which
//! ships with the NVIDIA display driver, so no SDK or toolkit is needed. It is
//! loaded at runtime: machines without an NVIDIA GPU simply report
//! `error.CudaUnavailable` and callers fall back to the CPU.
const std = @import("std");
const builtin = @import("builtin");

pub const supported = switch (builtin.os.tag) {
    .windows, .linux => builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .aarch64,
    else => false,
};

pub const Result = c_int;
pub const Device = c_int;
pub const Context = ?*opaque {};
pub const Module = ?*opaque {};
pub const Function = ?*opaque {};
pub const Stream = ?*opaque {};
pub const Event = ?*opaque {};
pub const DevPtr = u64;

const Api = struct {
    cuInit: *const fn (u32) callconv(.c) Result,
    cuDeviceGet: *const fn (*Device, c_int) callconv(.c) Result,
    cuDeviceGetCount: *const fn (*c_int) callconv(.c) Result,
    cuDeviceGetName: *const fn ([*]u8, c_int, Device) callconv(.c) Result,
    cuDeviceGetAttribute: *const fn (*c_int, c_int, Device) callconv(.c) Result,
    cuDeviceTotalMem_v2: *const fn (*usize, Device) callconv(.c) Result,
    cuDevicePrimaryCtxRetain: *const fn (*Context, Device) callconv(.c) Result,
    cuDevicePrimaryCtxRelease_v2: *const fn (Device) callconv(.c) Result,
    cuCtxSetCurrent: *const fn (Context) callconv(.c) Result,
    cuModuleLoadDataEx: *const fn (*Module, *const anyopaque, c_uint, ?[*]c_int, ?[*]?*anyopaque) callconv(.c) Result,
    cuModuleUnload: *const fn (Module) callconv(.c) Result,
    cuModuleGetFunction: *const fn (*Function, Module, [*:0]const u8) callconv(.c) Result,
    cuMemAlloc_v2: *const fn (*DevPtr, usize) callconv(.c) Result,
    cuMemFree_v2: *const fn (DevPtr) callconv(.c) Result,
    cuMemAllocHost_v2: *const fn (*?*anyopaque, usize) callconv(.c) Result,
    cuMemFreeHost: *const fn (?*anyopaque) callconv(.c) Result,
    cuMemcpyHtoDAsync_v2: *const fn (DevPtr, *const anyopaque, usize, Stream) callconv(.c) Result,
    cuMemcpyDtoHAsync_v2: *const fn (*anyopaque, DevPtr, usize, Stream) callconv(.c) Result,
    cuMemcpyDtoDAsync_v2: *const fn (DevPtr, DevPtr, usize, Stream) callconv(.c) Result,
    cuMemsetD32Async: *const fn (DevPtr, u32, usize, Stream) callconv(.c) Result,
    cuStreamCreate: *const fn (*Stream, c_uint) callconv(.c) Result,
    cuStreamDestroy_v2: *const fn (Stream) callconv(.c) Result,
    cuStreamSynchronize: *const fn (Stream) callconv(.c) Result,
    cuEventCreate: *const fn (*Event, c_uint) callconv(.c) Result,
    cuEventDestroy_v2: *const fn (Event) callconv(.c) Result,
    cuEventRecord: *const fn (Event, Stream) callconv(.c) Result,
    cuEventSynchronize: *const fn (Event) callconv(.c) Result,
    cuEventElapsedTime: *const fn (*f32, Event, Event) callconv(.c) Result,
    cuLaunchKernel: *const fn (Function, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Stream, ?[*]?*anyopaque, ?[*]?*anyopaque) callconv(.c) Result,
    cuGetErrorString: *const fn (Result, *?[*:0]const u8) callconv(.c) Result,
};

var api: Api = undefined;
var loaded: enum { no, yes, failed } = .no;
var load_mutex: std.atomic.Mutex = .unlocked;

// Windows: LoadLibrary/GetProcAddress from kernel32 (std.DynLib doesn't cover Windows).
extern "kernel32" fn LoadLibraryExA(name: [*:0]const u8, file: ?*anyopaque, flags: u32) callconv(.winapi) ?*anyopaque;
/// nvcuda.dll is installed by the NVIDIA driver into System32; look only
/// there, never in the current folder or on PATH (a planted copy).
const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x800;
extern "kernel32" fn GetProcAddress(module: *anyopaque, name: [*:0]const u8) callconv(.winapi) ?*anyopaque;

fn loadApi() !void {
    while (!load_mutex.tryLock()) std.atomic.spinLoopHint();
    defer load_mutex.unlock();
    switch (loaded) {
        .yes => return,
        .failed => return error.CudaUnavailable,
        .no => {},
    }
    errdefer loaded = .failed;
    if (!supported) return error.CudaUnavailable;
    if (builtin.os.tag == .windows) {
        const lib = LoadLibraryExA("nvcuda.dll", null, LOAD_LIBRARY_SEARCH_SYSTEM32) orelse return error.CudaUnavailable;
        inline for (@typeInfo(Api).@"struct".field_names) |name| {
            const p = GetProcAddress(lib, name) orelse return error.CudaUnavailable;
            @field(api, name) = @ptrCast(@alignCast(p));
        }
    } else {
        var lib = std.DynLib.open("libcuda.so.1") catch return error.CudaUnavailable;
        inline for (@typeInfo(Api).@"struct".field_names) |name| {
            @field(api, name) = lib.lookup(@FieldType(Api, name), name) orelse return error.CudaUnavailable;
        }
    }
    try check(api.cuInit(0));
    loaded = .yes;
}

pub fn check(r: Result) !void {
    if (r == 0) return;
    if (loaded == .yes) {
        var msg: ?[*:0]const u8 = null;
        if (api.cuGetErrorString(r, &msg) == 0) {
            if (msg) |m| std.log.debug("cuda error {d}: {s}", .{ r, m });
        }
    }
    return switch (r) {
        2 => error.CudaOutOfMemory,
        100 => error.CudaNoDevice,
        else => error.CudaError,
    };
}

/// One GPU, its primary context, and a stream all work is queued on.
pub const Gpu = struct {
    dev: Device,
    ctx: Context,
    stream: Stream,
    name: [128]u8 = @splat(0),
    total_mem: usize = 0,
    sm_count: u32 = 0,

    pub fn open(ordinal: c_int) !Gpu {
        try loadApi();
        var n: c_int = 0;
        try check(api.cuDeviceGetCount(&n));
        if (ordinal >= n) return error.CudaNoDevice;
        var g: Gpu = .{ .dev = 0, .ctx = null, .stream = null };
        try check(api.cuDeviceGet(&g.dev, ordinal));
        try check(api.cuDevicePrimaryCtxRetain(&g.ctx, g.dev));
        errdefer _ = api.cuDevicePrimaryCtxRelease_v2(g.dev);
        try check(api.cuCtxSetCurrent(g.ctx));
        try check(api.cuStreamCreate(&g.stream, 1)); // non-blocking w.r.t. the legacy stream
        _ = api.cuDeviceGetName(&g.name, g.name.len - 1, g.dev);
        _ = api.cuDeviceTotalMem_v2(&g.total_mem, g.dev);
        var sms: c_int = 0;
        _ = api.cuDeviceGetAttribute(&sms, 16, g.dev); // MULTIPROCESSOR_COUNT
        g.sm_count = @intCast(@max(sms, 1));
        return g;
    }

    pub fn close(g: *Gpu) void {
        g.makeCurrent();
        _ = api.cuStreamDestroy_v2(g.stream);
        _ = api.cuDevicePrimaryCtxRelease_v2(g.dev);
        g.* = undefined;
    }

    pub fn deviceName(g: *const Gpu) []const u8 {
        return std.mem.sliceTo(&g.name, 0);
    }

    /// CUDA contexts are per-thread; call before using the GPU from a new thread.
    pub fn makeCurrent(g: *const Gpu) void {
        _ = api.cuCtxSetCurrent(g.ctx);
    }

    pub fn alloc(_: *const Gpu, bytes: usize) !DevPtr {
        var p: DevPtr = 0;
        try check(api.cuMemAlloc_v2(&p, @max(bytes, 1)));
        return p;
    }
    pub fn free(_: *const Gpu, p: DevPtr) void {
        if (p != 0) _ = api.cuMemFree_v2(p);
    }

    /// Page-locked host memory: the GPU can DMA straight from it, so copies
    /// run at full PCIe speed and can overlap with CPU work.
    pub fn allocPinned(_: *const Gpu, bytes: usize) ![]align(4096) u8 {
        var p: ?*anyopaque = null;
        try check(api.cuMemAllocHost_v2(&p, bytes));
        const ptr: [*]align(4096) u8 = @ptrCast(@alignCast(p orelse return error.CudaOutOfMemory));
        return ptr[0..bytes];
    }
    pub fn freePinned(_: *const Gpu, mem: []align(4096) u8) void {
        _ = api.cuMemFreeHost(mem.ptr);
    }

    pub fn upload(g: *const Gpu, dst: DevPtr, src: []const u8) !void {
        try check(api.cuMemcpyHtoDAsync_v2(dst, src.ptr, src.len, g.stream));
    }
    pub fn download(g: *const Gpu, dst: []u8, src: DevPtr) !void {
        try check(api.cuMemcpyDtoHAsync_v2(dst.ptr, src, dst.len, g.stream));
    }
    pub fn copy(g: *const Gpu, dst: DevPtr, src: DevPtr, bytes: usize) !void {
        try check(api.cuMemcpyDtoDAsync_v2(dst, src, bytes, g.stream));
    }
    /// Sets `count` 32-bit words at `dst` to `value` (0 also zeroes f32s).
    pub fn memset32(g: *const Gpu, dst: DevPtr, value: u32, count: usize) !void {
        try check(api.cuMemsetD32Async(dst, value, count, g.stream));
    }
    pub fn sync(g: *const Gpu) !void {
        try check(api.cuStreamSynchronize(g.stream));
    }

    pub fn loadModule(_: *const Gpu, ptx: [:0]const u8) !Module {
        var log: [4096]u8 = @splat(0);
        var opts = [_]c_int{ 5, 6 }; // ERROR_LOG_BUFFER, ERROR_LOG_BUFFER_SIZE_BYTES
        var vals = [_]?*anyopaque{ &log, @ptrFromInt(log.len) };
        var m: Module = null;
        check(api.cuModuleLoadDataEx(&m, ptx.ptr, opts.len, &opts, &vals)) catch |err| {
            std.log.err("PTX load failed: {s}", .{std.mem.sliceTo(&log, 0)});
            return err;
        };
        return m;
    }
    pub fn unloadModule(_: *const Gpu, m: Module) void {
        _ = api.cuModuleUnload(m);
    }
    pub fn function(_: *const Gpu, m: Module, name: [:0]const u8) !Function {
        var f: Function = null;
        check(api.cuModuleGetFunction(&f, m, name)) catch |err| {
            std.log.err("kernel '{s}' missing from PTX", .{name});
            return err;
        };
        return f;
    }

    /// Queues `f<<<grid, block>>>(args...)`. Each arg must be a u32, i32, f32 or DevPtr.
    pub fn launch(g: *const Gpu, f: Function, grid: [3]u32, block: u32, args: anytype) !void {
        const info = @typeInfo(@TypeOf(args)).@"struct";
        // Copy each value into runtime storage (tuple fields may be comptime).
        var store: [info.field_names.len]u64 = undefined;
        var ptrs: [info.field_names.len]?*anyopaque = undefined;
        inline for (info.field_names, info.field_types, 0..) |name, T, i| {
            comptime switch (T) {
                u32, i32, f32, u64 => {},
                else => @compileError("unsupported kernel arg type " ++ @typeName(T)),
            };
            @as(*T, @ptrCast(&store[i])).* = @field(args, name);
            ptrs[i] = &store[i];
        }
        try check(api.cuLaunchKernel(f, grid[0], grid[1], grid[2], block, 1, 1, 0, g.stream, &ptrs, null));
    }

    pub fn event(_: *const Gpu) !Event {
        var e: Event = null;
        try check(api.cuEventCreate(&e, 0));
        return e;
    }
    /// An event without timing data: cheaper to record and wait on.
    pub fn fence(_: *const Gpu) !Event {
        var e: Event = null;
        try check(api.cuEventCreate(&e, 2)); // DISABLE_TIMING
        return e;
    }
    pub fn destroyEvent(_: *const Gpu, e: Event) void {
        _ = api.cuEventDestroy_v2(e);
    }
    pub fn record(g: *const Gpu, e: Event) !void {
        try check(api.cuEventRecord(e, g.stream));
    }
    pub fn wait(_: *const Gpu, e: Event) !void {
        try check(api.cuEventSynchronize(e));
    }
    /// Milliseconds between two recorded (timing) events.
    pub fn elapsed(_: *const Gpu, a: Event, b: Event) f32 {
        var ms: f32 = 0;
        _ = api.cuEventElapsedTime(&ms, a, b);
        return ms;
    }
};

test "driver library loads from System32 only (skipped without an NVIDIA driver)" {
    if (builtin.os.tag != .windows) return error.SkipZigTest;
    // A DLL in the current folder is not found with the System32-only search
    // (the default search order would load it).
    const io = std.testing.io;
    const name = "stt-planted-test.dll";
    const exe = std.Io.Dir.cwd().readFileAlloc(io, "C:\\Windows\\System32\\version.dll", std.testing.allocator, .limited(16 << 20)) catch return error.SkipZigTest;
    defer std.testing.allocator.free(exe);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = name, .data = exe });
    defer std.Io.Dir.cwd().deleteFile(io, name) catch {};
    try std.testing.expect(LoadLibraryExA(name, null, LOAD_LIBRARY_SEARCH_SYSTEM32) == null);
    // The real driver, where there is one, loads from System32.
    loadApi() catch return error.SkipZigTest;
}
