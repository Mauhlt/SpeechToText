# stt: speech-to-text in pure Zig, on the GPU

An encoder-decoder speech transformer written from scratch in Zig: log-mel
front end, convolutional + transformer audio encoder, KV-cached text decoder,
BPE tokenizer, and a hand-written backward pass for personal fine-tuning.
Inference **and** training run on an NVIDIA GPU, with kernels written in Zig.

It runs on the home desktop (`stt.Home`, NVIDIA GPU, which also trains) and
on a Snapdragon X laptop (`stt.Local`, on the laptop's NPU, so laptop audio
never leaves the laptop). [Portal](../Portal) captures the mic, carries audio
in and text out, and moves sync data between them; stt has no server, mic
capture or UI of its own.

No third-party code. The outside pieces are a pretrained weight file (data),
the NVIDIA driver (`nvcuda.dll` / `libcuda.so`, part of the display driver),
and the sibling [Audio](../Audio) library for decoding WAV / MP3 / QOA.

Needs Zig 0.18.0-dev.4, an NVIDIA GPU (GTX 10xx / Pascal or newer) on Windows
or Linux, and `../Audio` checked out next to this folder. Builds in
ReleaseFast by default (`zig build`).

## Setup

The code reads weight files in the ggml container format. Compatible
pretrained English weights:

```sh
zig build
curl -L -o models/ggml-base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
# smaller/faster: ggml-tiny.en.bin · more accurate: ggml-small.en.bin (f16/f32 only, not quantized)
```

> Weights attribution: those files are OpenAI's Whisper model weights, MIT-licensed
> (Copyright (c) 2022 OpenAI). Keep that notice if you redistribute them.

## Using it from Portal

`stt.Home` is the one object Portal needs:

```zig
const stt = @import("stt"); // dependency in build.zig.zon

const home = try stt.Home.open(gpa, io, .{
    .model_path = "models/ggml-base.en.bin",
    .profile_dir = profile_dir,
    // optional: .schedule = .{ .window = .{ .start_minute = 2 * 60, .end_minute = 6 * 60 }, .min_new_samples = 20 },
});
defer home.close();
home.setListener(.{ .ctx = portal, .onEvent = onTrainingEvent }); // optional
home.setWakeup(.{ .ctx = portal, .onWakeup = scheduleTick });      // when to call tick

const r = try home.transcribe(gpa, audio_file_bytes, "iphone"); // WAV/MP3/QOA -> { id, text }
defer gpa.free(r.text);
_ = try home.feedback(r.id, corrected_text, true);               // see "Learning" below

// in Portal's timer, at the deadline it was given:
const next = home.tick(); // ?i64 unix ms of the next deadline; null = nothing pending
```

**No polling.** `tick` only does work that has a time attached: auto-confirming
a transcript whose silence period ran out, and starting training when the
window opens. It returns when it next needs to run. The wakeup hook reports
the deadline whenever it changes (after a transcription, feedback, an import,
a schedule change, or a finished training run), so Portal can sleep until
exactly then. Training runs finish on their own thread; no tick is needed for that.

All `Home` methods are thread-safe. Sync between devices and home
(`importBundle`, `exportUpdate`) goes through `Home` too.

## GPU or NPU backend

The decoding loop (`src/transcriber.zig`) drives either engine through the
same encode / step / pick interface:

```zig
var tr = try stt.Transcriber.init(gpa, &model, .{});                               // .backend = .cuda (default): NVIDIA GPU
var tr = try stt.Transcriber.init(gpa, &model, .{ .backend = .{ .npu = .{} } });   // Snapdragon X NPU via QNN
var tr = try stt.Transcriber.init(gpa, &model, .{ .backend = .{ .coreml = .{ .cache_dir = caches } } }); // iPhone / Mac: Core ML (Neural Engine)
```

`npu` options: `qnn_dir` (folder with the QNN backend DLLs, null = search),
`cache_dir` (compiled-graph cache; `stt.Local` defaults it to its profile dir)
and `backend = .htp` (the NPU) or `.cpu` (QNN's CPU backend, for development).
There is no CPU inference path of stt's own and no CPU compute threads.
`Home` always uses the GPU. On Windows on Arm the cuda backend is compiled out
(returns `error.CudaUnavailable`) and only `stt-local` is built.

`coreml` is built for iOS / macOS targets only: TranslateCoreML is a lazy
dependency, so Windows and Linux builds never fetch or compile it (and
`.coreml` does not compile there). `-Dcoreml` includes it on other targets,
where it runs on TranslateCoreML's host reference interpreter:
`zig build test -Dcoreml` checks the programs against the GPU. Options
(`cache_dir`, `compute_units`, `runtime`, ...) are in `src/coreml/engine.zig`;
Apple-target checks and the iPhone test app are in `src/coreml/dev`, and the
Mac / iPhone checklist is `docs/mac-session.md`.

> Status: the NPU engine (`src/npu/engine.zig`) is being written; until it
> lands, opening the NPU backend returns `error.NpuNotImplemented`.

## The laptop: `stt.Local`

```zig
const lap = try stt.Local.open(gpa, io, .{
    .model_path = model_path,       // the stt-model data package (read only)
    .profile_dir = data_dir,        // %LOCALAPPDATA%\Portal\sync\data\stt-local\ (the only place written)
    .adapter_path = adapter_path,   // optional: the stt-adapter data package (read only)
    .npu = .{ .qnn_dir = null },    // cache_dir defaults to profile_dir
    // .backend = .cuda             // run the same flow on an NVIDIA GPU (tests)
});
defer lap.close();                                     // confirms pending transcripts
lap.setWakeup(.{ .ctx = portal, .onWakeup = scheduleTick });

const r = try lap.transcribePcm(gpa, samples_16k_f32); // or lap.transcribe(gpa, wav_bytes)
_ = try lap.feedback(r.id, corrected_text, true);
const next = lap.tick();                               // at the deadline: silence = correct

// sync, device side:
if (try lap.exportBundle(gpa, stt.sync.default_bundle_bytes)) |b| { ... send, then lap.commitExport(b.through) }
_ = try lap.importUpdate(gpa, update_bytes);           // new adapter switched in right away, plus rules/vocab
```

Same rules as `Home`: one lock makes it thread-safe, no polling (the only
timed work is the 60 s silence auto-confirm), and it starts no threads.
`status()` reports the backend, adapter version, learning flag and samples
waiting to go home. The `adapter_path` package may be a raw `adapter.bin`
(used when the profile has no adapter of its own or `adapter_version` is
newer) or an update file from `Home.exportUpdate` (imported if newer). If the
engine fails, transcription returns `error.BackendLost` from then on.

### `stt-local`: the program Portal runs

Portal installs `stt-local.exe` read-only under
`%LOCALAPPDATA%\Portal\sync\packages\stt-local\<ver>\` (model under
`packages\stt-model\<ver>\`, adapter under `packages\stt-adapter\<ver>\`) and
runs it as a child process:

```
stt-local --model <path> --profile <dir> [--adapter <path>] [--adapter-version <n>]
          [--qnn <dir>] [--backend htp|cpu|cuda] [--device-name <name>]
```

It speaks JSON lines on stdin/stdout, protocol version 1 (agreed with Portal):

- stdout carries only protocol lines: UTF-8, one JSON object per line, at
  most 8 MiB per line. Logs go to stderr.
- The first line is
  `{"event":"ready","proto":1,"version":"0.1.0","backend":"htp|cpu|cuda","model":"<model file name>","adapter_version":N}`.
- Every request has a `"rid"` (u32) that its reply echoes. Replies may come
  out of order (this version answers in order). Events have no rid.
- Errors: `{"rid":N,"error":"<code>","message":"..."}`, code one of
  `bad_request`, `bad_audio`, `too_long`, `busy`, `backend_lost`,
  `unknown_id`, `not_found`, `internal`. Malformed JSON gets rid 0.
  `backend_lost`: the NPU/GPU engine failed for good; restart the process.
  Bad input never crashes it.

| request | reply |
|---|---|
| `{"rid":1,"op":"transcribe","pcm_s16le_b64":"..."}` (raw 16 kHz mono s16le, preferred) or `"wav_b64"` (WAV/MP3/QOA file) | `{"rid":1,"id":7,"text":"..."}`; id 0 = nothing heard. Over 120 s: `too_long` |
| `{"rid":2,"op":"feedback","id":7,"text":"...","ok":true}` | `{"rid":2,"outcome":"learned","sample_id":3,"new_rules":1,"new_words":2}` or `"outcome":"discarded"`; an expired id: `unknown_id` |
| `{"rid":3,"op":"export_bundle","max_bytes":8388608}` | `{"rid":3,"b64":"...","through":12,"count":4,"version":2}`, or `"b64":null,"count":0` when nothing is pending (bundles are capped at 5 MiB so the line fits) |
| `{"rid":4,"op":"commit_export","through":12}` | `{"rid":4,"ok":true}` |
| `{"rid":5,"op":"import_update","b64":"..."}` | `{"rid":5,"version":3,"adapter_version":3}` |
| `{"rid":6,"op":"set_learning","on":false}` | `{"rid":6,"ok":true}` |
| `{"rid":7,"op":"status"}` | `{"rid":7,"proto":1,"version":..,"backend":..,"device":..,"model":..,"adapter_version":..,"profile_version":..,"learning":..,"pending_export":..,"deadline_unix_ms":..}` |

Events: `{"event":"wakeup","deadline_unix_ms":N|null}` whenever the tick
deadline changes. stt-local calls tick itself at that time (a timer thread
sleeps until exactly then; no fixed-interval polling), so Portal needs no
timer of its own.

Lifetime and sandbox: close stdin to stop it; it confirms pending transcripts
and exits 0 well within 2 s (about 50 ms measured). No network, sockets or
child processes; bundles leave only through stdout; it writes only under
`--profile`.

## Learning from feedback

stt never asks the user anything. Your app (via Portal) sends one of three responses:

| response | how | result |
|---|---|---|
| fix | `feedback(id, corrected_text, true)` | learned: phrase rule + vocabulary + training sample |
| silence | no feedback within 60 s (or 64 newer transcripts, or shutdown) | saved as a confirmed sample |
| nothing | `setLearning(false)` | no audio kept |

`feedback(id, _, false)` discards a transcript that was wrong when the right
text isn't known. Everything learned lives in the profile folder:

- `rules.txt`: phrase fixes like `puresig => pure Zig`, applied right away
- `vocab.txt`: your words, fed to the decoder as a prompt
- `samples/`: (audio, correct text) pairs, tagged with the device they came from
- `adapter.bin`: the personal LoRA adapter, plus every version in `adapters/vNNNN.bin`

## Training on the GPU, on a schedule

Training learns low-rank (LoRA) adapters on every decoder layer with a
hand-written backward pass and the **Muon** optimizer (Newton–Schulz-
orthogonalized momentum; Adam is also available), all on the GPU. It took
about 1 s for 6 samples × 3 epochs on base.en / RTX 4060. The new adapter is
merged into the transcription weights on the GPU as soon as it finishes.

Training uses the GPU hard, so other GPU work (transcription too) is slower
while it runs. It only **starts** automatically:

- inside the daily window (local time, default **02:00–06:00**), and
- once `min_new_samples` (default 20) samples have arrived since the last version.

`home.trainNow()` starts it immediately; `home.cancelTraining()` stops it
(the previous adapter stays). `home.setSchedule(...)` changes the window
(`window = null` means manual only).

**So Portal can warn you**, `home.trainingStatus()` returns:

| field | meaning |
|---|---|
| `state` | `idle`, `waiting_for_window` (enough samples, window not open yet), `running` |
| `pending_samples` / `min_new_samples` | progress toward the next run |
| `next_start_unix` | when the window opens (if waiting) |
| `started_unix`, `progress` (0–1), `eta_seconds`, `epoch`/`epochs` | the current run |
| `version` | adapter version transcription is using |
| `last_finished_unix`, `last_loss`, `last_samples`, `last_error` | the previous run |

The optional listener receives `started`, `progress`, `finished`, `failed`
and `canceled` events. `started` comes from the call that started the run;
the rest come from the training thread. The listener and the wakeup hook
should return quickly and not call back into `Home`.

The training run uses its own copy of the weights in VRAM (base.en: about
140 MB more, plus activations for the longest sample).

## Sync between devices

One home profile owns the single shared adapter. The Snapdragon laptop runs
`stt.Local` on its NPU and ships its feedback home during an authenticated
Portal sync (`Local.exportBundle` / `commitExport` / `importUpdate`, or the
matching stt-local ops); devices that don't run stt (iPhone) just send
audio to `Home.transcribe`. The lower-level calls:

```zig
// device, on each sync:
while (try stt.sync.exportBundle(gpa, &profile, "laptop", stt.sync.default_bundle_bytes)) |b| {
    defer b.deinit(gpa);
    try send(b.bytes, profile.version);            // home stores it durably, then acks
    try stt.sync.commitExport(&profile, b.through); // deletes the delivered samples locally
}
if (try receiveUpdate()) |bytes| _ = try stt.sync.importUpdate(gpa, &profile, bytes);

// home:
_ = try home.importBundle(gpa, bundle_bytes);
if (try home.exportUpdate(gpa, device_version)) |u| reply(u);
```

Sizes: one sample is at most about 1 MB (30 s of 16-bit audio). Bundles are
size-capped and chunkable (8 MB by default). An update is the adapter (about
1.1 MB tiny, 2.2 MB base, 6.5 MB small) plus the rules and vocab text. Devices
must use the same base model file as home.

**Storage cap:** once home samples exceed 50 GB (`StoragePolicy.max_bytes`),
the oldest audio that is already trained in is deleted after a training run
until usage is under 80%, keeping every 10th sample as a replay set.
Transcript text and metadata are kept. Unpairing a device never deletes its
samples.

## CLI (for testing on the desktop)

```sh
stt transcribe clip.wav            # WAV / MP3 / QOA; uses your profile's adapter and rules
stt bench clip.wav --runs 5        # timings: load, GPU setup, encoder / decode split
stt train                          # train the adapter now (ignores the schedule)
stt status                         # samples waiting, adapter version, schedule
```

Options: `--model <path>`, `--prompt "Zig, comptime"`, `--profile <dir>`,
`--no-profile`, and for `train`: `--epochs`, `--lr`, `--optimizer muon|adam`.

## How it uses the GPU

The kernels in `src/gpu/kernels.zig` are Zig, compiled to PTX by Zig's own
NVPTX backend at build time and embedded in the library. The driver
JIT-compiles them for the installed card on first run (and caches the result).

Measured on an RTX 4060, base.en (the CPU column is the old CPU version on an i5-14400F, for scale):

| | old CPU | GPU |
|---|---|---|
| 5.6 s clip | 2140 ms | 45 ms (124x realtime) |
| 47.7 s clip (2 windows) | 9870 ms | 189 ms (252x realtime) |

Getting data to the GPU:

- **Weights** stay f16 from the file through to the GPU (half the bytes of
  f32, no conversion), uploaded once into one device allocation through two
  pinned staging buffers that alternate. Model load is about 90 ms, the upload about 25 ms.
- **Per window**, only the raw audio samples go up (64 KB per second of
  audio, via pinned memory). Padding, STFT, log-mel, convolutions, encoder
  and cross-attention keys/values are computed on the GPU and never leave it.
- **Per token**, the GPU picks the token itself and returns 12 bytes instead
  of the 200 KB logit vector.
- **Training** uploads each sample's audio and tokens once; the forward and
  backward passes, gradient clipping and the optimizer all stay on the GPU.

## Layout

| file | what |
|------|------|
| `src/home.zig`          | `Home`: what Portal uses (transcribe, feedback, sync, scheduled training) |
| `src/local.zig`         | `Local`: the laptop (transcribe on the NPU, feedback, device-side sync) |
| `src/local_main.zig`    | `stt-local`: Local behind the JSON-lines protocol for Portal |
| `src/npu/engine.zig`    | NPU engine (Snapdragon X Hexagon via QNN) |
| `src/coreml/engine.zig` | Core ML engine (iPhone / Mac Neural Engine), programs from `coreml/net.zig`, host log-mel `coreml/frontend.zig` |
| `src/schedule.zig`      | training window, status and events |
| `src/training.zig`      | one training run: collect samples, train, record the version |
| `src/service.zig`       | transcription + feedback window for Portal clients |
| `src/transcriber.zig`   | greedy decoding over the GPU, NPU or Core ML engine |
| `src/gpu/kernels.zig`   | GPU kernels (Zig → PTX): matmul, flash attention, log-mel, backward pass, Muon/Adam |
| `src/gpu/engine.zig`    | GPU weights, uploads, encoder/decoder, adapter merge |
| `src/gpu/trainer.zig`   | LoRA forward/backward and optimizer steps on the GPU |
| `src/gpu/cuda.zig`      | runtime binding to the NVIDIA driver API |
| `src/model.zig`         | weight-file loader |
| `src/adapter.zig`       | LoRA adapter data and file format |
| `src/audio.zig`         | audio decode (via ../Audio), resample to 16 kHz mono |
| `src/mel.zig`           | log-mel parameters and DFT table |
| `src/bpe.zig`           | BPE tokenizer |
| `src/profile.zig`       | feedback → rules, vocabulary, samples |
| `src/window.zig`        | the pending-feedback window (silence = correct) |
| `src/sync.zig`          | device ↔ home bundles, versioned updates, storage cap |
