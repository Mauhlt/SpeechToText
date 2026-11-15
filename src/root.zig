//! stt: speech-to-text in pure Zig: an encoder-decoder transformer written
//! from scratch, run (and personally fine-tuned) on an NVIDIA GPU.
//!
//! The home desktop, driven by Portal:
//!
//!     var home = try stt.Home.open(gpa, io, .{
//!         .model_path = "models/ggml-base.en.bin",
//!         .profile_dir = profile_dir,
//!     });
//!     defer home.close();
//!     const r = try home.transcribe(gpa, audio_file_bytes, "iphone"); // -> { id, text }
//!     _ = try home.feedback(r.id, corrected_text, true);
//!     const next_ms = home.tick(); // at the deadline from setWakeup: silence = correct, scheduled training
//!     if (home.trainingStatus().state == .running) warnUser();
//!
//! The Snapdragon X laptop (local, on the NPU; audio never leaves it):
//!
//!     var lap = try stt.Local.open(gpa, io, .{ .model_path = model, .profile_dir = data_dir });
//!     defer lap.close();
//!     const r = try lap.transcribePcm(gpa, samples_16k);       // -> { id, text }
//!     if (try lap.exportBundle(gpa, max)) |b| ...              // ship corrections home
//!     _ = try lap.importUpdate(gpa, update_bytes);             // new adapter from home
//!
//! Lower level: `Model` + `Transcriber` for plain transcription.
const std = @import("std");

pub const model = @import("model.zig");
pub const mel = @import("mel.zig");
pub const transcriber = @import("transcriber.zig");
pub const audio = @import("audio.zig");
pub const bpe = @import("bpe.zig");
pub const profile = @import("profile.zig");
pub const adapter = @import("adapter.zig");
pub const training = @import("training.zig");
pub const schedule = @import("schedule.zig");
pub const sync = @import("sync.zig");
pub const window = @import("window.zig");
pub const service = @import("service.zig");
pub const home = @import("home.zig");
pub const local = @import("local.zig");
/// Laptop NPU (Snapdragon X Hexagon) backend, through QNN.
pub const npu = struct {
    pub const engine = @import("npu/engine.zig");
};
/// iPhone / Mac backend: Core ML (Neural Engine), from ML Programs written in Zig.
/// Empty unless the build includes it (`transcriber.coreml_enabled`).
pub const coreml = if (transcriber.coreml_enabled) struct {
    pub const engine = @import("coreml/engine.zig");
    pub const net = @import("coreml/net.zig");
    pub const frontend = @import("coreml/frontend.zig");
    /// TranslateCoreML (MIL builder, package writer, reference interpreter, runtime).
    pub const lib = @import("coreml");
} else struct {};
pub const gpu = struct {
    pub const cuda = @import("gpu/cuda.zig");
    pub const engine = @import("gpu/engine.zig");
    pub const trainer = @import("gpu/trainer.zig");
};

pub const Model = model.Model;
pub const Transcriber = transcriber.Transcriber;
pub const Options = transcriber.Options;
pub const Profile = profile.Profile;
pub const Adapter = adapter.Adapter;
pub const Service = service.Service;
/// Everything the home desktop runs, in one object (what Portal uses).
pub const Home = home.Home;
/// The laptop: local transcription on the NPU, sync bundles home.
pub const Local = local.Local;

test {
    std.testing.refAllDecls(@This());
    _ = transcriber;
    _ = audio;
    _ = profile;
    _ = sync;
    _ = training;
    _ = bpe;
    _ = model;
    _ = adapter;
    _ = npu.engine;
    _ = gpu.engine;
    _ = window;
    _ = service;
    _ = schedule;
    _ = local;
    if (transcriber.coreml_enabled) {
        _ = coreml.frontend;
        _ = @import("coreml/tests.zig");
    }
    _ = @import("tests.zig");
}
