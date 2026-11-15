# Development notes

## Shape of the project (2026-11-11)

stt is an event-driven (no polling: `tick` returns its next deadline) library that Portal embeds on the home desktop
(`stt.Home`, NVIDIA GPU) and runs on the Snapdragon laptop (`stt-local` /
`stt.Local`, on the NPU). Portal moves audio in and text out; stt has no server, mic
capture, voice detection or CPU inference path anymore (all removed
2026-11-11). Audio decoding comes from the `Audio` library, a URL package
pinned in `build.zig.zon` (`zig build --fork=../Audio` builds against a local
checkout).

Tested on an NVIDIA RTX 40-series GPU (sm_89, 8 GB).

## Laptop: stt.Local and stt-local (2026-11-12)

- `transcriber.zig` drives `Engine = union(enum) { gpu, npu }`; `Options.backend`
  is `.cuda` (default, Home) or `.npu = npu.Options`. The GPU path's output and
  timing are unchanged (bench: same text, ~45 ms base.en).
- `local.zig` (`stt.Local`): Model + Profile + Transcriber + Service under one
  lock, like Home minus training. Only timed work: silence auto-confirm.
  `importUpdate` writes the adapter into the profile and calls
  `engine.setAdapter` at once. Writes only under `profile_dir` (the NPU graph
  cache defaults there). Tests run it with `backend = .cuda` on the desktop GPU.
- `local_main.zig` (`stt-local` exe): JSON-lines protocol v1, confirmed with
  Portal (see README). Requests are handled in order on the main thread; a
  timer thread only sleeps on a condition until the tick deadline (no compute,
  no polling). `busy` is reserved (never sent: requests are serialized) and
  `not_found` only maps FileNotFound. `set_learning` is an extra op.
  `src/local_protocol_test.zig` spawns the built exe and checks ready, rid
  echo, error codes, transcription and the EOF exit.
- aarch64-windows: `zig build -Dtarget=aarch64-windows` builds `stt-local`
  and the `stt` CLI; the NPU engine (and so TranslateNPU and the QAIRT SDK) is
  compiled only for this target (or with `-Dnpu`).

## NPU handover (2026-11-12)

**Resume here.** Paused so the laptop can be paired over WireGuard (via Portal)
and tested on the real NPU; the x86 HTP simulator is too slow to run the graphs.
At pause: `zig build`, `zig build test` and `-Dtarget=aarch64-windows` all pass;
QnnCpu matches the GPU transcript; HTP graphs compile but are unrun. Next:
1. Portal delivers and runs `npu-probe`; bring `npu-probe.json` back.
2. On the laptop: `npu-dev --backend htp --no-gpu --mel`, then full tiny.en,
   comparing with the GPU transcript (commands below).
3. Check FP16 numerics; try rpcmem shared buffers (avoid copying cross K/V each step).
4. Tell Portal when the `stt-local` package is ready to publish.

Files: `src/npu/engine.zig` (QNN session, cache, interface), `src/npu/net.zig`
(encoder and decoder-step graphs), `src/npu/graph.zig` (op builder),
`src/npu/dev/` (separate build: the `npu-dev` check program). The helpers in
TranslateNPU's `src/qnn.zig` (`scalarParam`, `tensorParam`, `findBackend`)
are committed there; stt pins TranslateNPU in `build.zig.zon`.

Design: two graphs built op by op from QNN built-ins, FP16 on the HTP (FP32 on
QnnCpu). Encoder: reflect-padded audio [3002][160] -> frames (StridedSlice+Concat)
-> windowed DFT (2x FullyConnected) -> power -> mel FC -> Log -> clamp max-8 ->
(x+4)/4 -> Conv2d 1x3 (conv1/conv2) + GELU -> blocks -> ln_post -> cross K/V per
layer [heads][1500][64]. Decoder step: token/pos (i32, Gather embeddings), self
KV cache [heads][448][64] in/out, mask, cross K/V, logits + ban mask -> Argmax,
LogSoftmax (pick log-prob = ReduceMax, no-speech = Gather). A `step(.., true)` is
executed when `pick()` supplies the ban list. GELU = tanh form via Sigmoid (as
the GPU); attention scale folded into q. Compiled contexts are cached in
`cache_dir` (key: backend build/arch, model hash, adapter, source of net.zig
and graph.zig).
Ops used: FullyConnected, MatMul, Conv2d, LayerNorm, Softmax, LogSoftmax,
Argmax, ReduceMax, Gather, Reshape, Transpose, Concat, StridedSlice,
ElementWiseAdd/Multiply/Maximum/Log, Sigmoid. No custom ops; nothing decomposed
for lack of support, as far as the simulator can tell.

Validated on this desktop:
- QnnCpu.dll (SDK x86 lib) vs the GPU, tiny.en + hello.wav: log-mel max error
  0.00000, cross K/V of layer 0 max abs error 5e-5 / 3e-5, transcript identical
  (" Hello World. ... Piersig."), also with the in-graph pick. setAdapter
  (random LoRA) rebuilds the decoder, and going back to base gives the same text.
  QnnCpu speed: encode 450-550 ms, 140-210 ms per decoder step (a reference
  backend, not representative).
- QnnHtp.dll x86 simulator (reports V68): both full graphs PASS HTP validation
  and FP16 finalize (encoder about 12 s, decoder about 4 s for tiny), and context
  binaries are saved to and reloaded from the cache (enc 20 MB, dec 54 MB).
  Executing them on the simulator is impractically slow (the log-mel graph alone
  did not finish in 25 min), so **HTP FP16 numerics are NOT validated**.

Open:
- "graph tensor x has no host buffer": a stale cached decoder binary from
  before the in-graph embedding was loaded. The cache key covered the model,
  backend and adapter but not the graph definition. Fixed: the key now hashes the
  source of net.zig and graph.zig, and a cached graph whose tensors don't bind
  falls back to a rebuild. Not yet re-run after the fix (the hybrid run was
  killed). Old cache files can simply be deleted.
- `npu-dev --hybrid N` (encoder on QnnCpu, N decoder picks on the HTP, compared
  token by token) was written but has not completed a run.
- FP16 risks only the real NPU can answer: log-mel dynamic range (audio scaled
  by 2^k on the host, undone in the log domain; subnormal/FTZ handling decides
  the bottom ~1 decade), LayerNorm of large residuals, Argmax ties on FP16
  logits, Log(0) behaviour (the floor is clamped with Maximum).
- Speed: inputs are QNN client buffers, so each step copies the cross K/V
  (tiny about 9 MB, base about 18 MB in FP16) and the self cache into the HTP.
  Next optimisation: rpcmem shared buffers + memRegister, and an HTP perf/power
  vote (burst). Neither has been tried.

Still on the host CPU (single-threaded, no worker threads), and why:
- Reflect-padding audio into the input buffer, and choosing its power-of-two
  scale (one pass of window-weighted |x| sums over 3000 frames, about 1 ms). The
  scale must be known before the data enters FP16.
- Attention-mask row (449 values per step) and the vocabulary ban mask (written
  only when the ban list changes, twice per window). Building them in-graph
  needs int32 compare/select, whose HTP support was not verified.
- Copying each step's new K/V row into the cache buffer. Graphs are stateless,
  and the cache goes back in as an input.
- LoRA merge at setAdapter (W += scale·B·A, once). Static weights must be host
  memory when the graph is built. A separate QNN merge graph would cost a
  compile that is larger than the merge itself.
- QnnCpu.dll itself may use its own threads (dev backend only).

On the laptop (Snapdragon X, Windows on Arm):
- Build `zig build -Dtarget=aarch64-windows` (stt-local). For npu-dev, build
  natively on the laptop: `cd src\npu\dev && zig build`.
- DLLs expected (installed by the NPU driver, never bundled): `QnnHtp.dll` (or
  `QairtHtp.dll`), plus its `QnnHtpV73Stub.dll` / `libQnnHtpV73Skel.so` /
  `libqnnhtpv73.cat` beside it, and `QnnHtpPrepare.dll` for online graph
  compilation. Search order: `--qnn <dir>` (2 levels deep), the exe folder,
  PATH, `%SystemRoot%\System32`, `System32\DriverStore\FileRepository` (3 deep)
  and `C:\Qualcomm` (6 deep). Run `npu-probe` (TranslateNPU) first to see what is
  installed.
- `src\npu\dev\zig-out\bin\npu-dev.exe --backend htp --no-gpu --cache <dir>`
  (run from the SpeechToText folder; add `--qnn <dir>` if it isn't found;
  `--prepare-only` compiles only; `--hybrid 6` compares against QnnCpu if
  QnnCpu.dll is available; `--mel` checks the front end; `--adapter` tests
  setAdapter).
- `stt-local --backend htp --model models\ggml-base.en.bin --profile <dir>
  [--qnn <dir>]`. See `stt-local --help` for the exact flags.

First thing to try next: on the laptop, `npu-dev --backend htp --no-gpu --mel`
then the full run on tiny.en. Compare the transcript with the GPU's (" Hello
World. This is a test of speech recognition written in Piersig."), and note
the encode and per-step timings.

## GPU

Kernels in `src/gpu/kernels.zig` are Zig, compiled by Zig's NVPTX backend
(`nvptx64-cuda`, sm_60 + PTX 6.3) in `build.zig` and embedded.
`src/gpu/cuda.zig` loads the driver API (`nvcuda.dll` / `libcuda.so`) at
runtime, with no CUDA toolkit or cuBLAS. Gotchas:
- `@exp` and `@log` have no libcall on nvptx; use the inline-asm `ex2.approx` / `lg2.approx` helpers.
- Shared-memory tiles need `align(16)` plus a pointer cast to get `ld.shared.v4`.
- `@min(comptime, runtime)` narrows the integer type, so annotate `usize` where it's multiplied.

One generic batched, strided `gemm` (f16 or f32 weights) does every matrix
product: inference A·Wᵀ, training's A·B / Aᵀ·B, and per-head attention.

### Measured (base.en)

| | GPU |
|---|---|
| 5.6 s clip | 45 ms (encoder+mel 24, cross-KV 2, decode 18–21 ms) |
| 47.7 s clip | 189 ms |
| model load / weight upload | ~90 ms / ~25 ms (f16 end to end) |
| training, 6 samples × 3 epochs | ~1 s |

### Training checks done

- The GPU LoRA gradients matched the old CPU trainer (removed): loss 0.37204
  vs 0.37186; gradient cosine 0.999998; relative error 0.2% overall and
  ≤0.6% for every layer and slot (from f16 weights and GPU mel).
- `src/tests.zig` checks the GPU gradient against finite differences on
  every run, plus a full `Home` cycle (transcribe → fix → scheduled
  training → adapter in use).
- A single-sample run fixes "Puresig" → "pure Zig" from the adapter alone (no rules).

## Training schedule

`schedule.zig`: training starts automatically only inside a daily
local-time window (default 02:00–06:00) once `min_new_samples` (20) have
arrived; `trainNow` starts one at once. Status and events
(`started`/`progress`/`finished`/`failed`/`canceled`) let Portal warn the user
while it runs. Training runs on a background thread with its own Engine (a
second weight copy in VRAM). It reads only a snapshot job (`training.collect`)
and never touches the Profile directly, so `Home` serializes all Profile
access under one lock.

## Next steps (not done)

1. **CUDA graphs for the decoder**: it's launch-bound (~94 launches, ~0.85 ms
   per token). Keep token/pos in device memory so one captured graph can replay.
2. **Tensor-core GEMM** (`mma.sync` f16) for the encoder (~24 ms) and training.
3. Training memory: attention probabilities are stored for the backward
   pass (H×L×1500 per layer). That's fine for dictation-length samples; large
   models with 448-token samples would need recomputation (flash-style backward).
4. Linux local time: `schedule.localMinute` reads local time on Windows only (UTC elsewhere).
5. iOS: the old CLI hit a Zig std bug (`Io.Threaded`). Moot now (GPU-only, desktop).

## iPhone: Core ML backend (2026-11-12)

Goal: stt on the iPhone's Neural Engine for the native Portal app (Apple's
SFSpeechRecognizer stays the fallback, in Portal). The ANE is reachable only
through Core ML, so `Backend.coreml` builds two Core ML ML Programs in Zig
(no Python / coremltools) with [TranslateCoreML](https://github.com/Mauhlt/TranslateCoreML) (MIL builder, protobuf
writer, .mlpackage, Core ML runtime via the Objective-C runtime, opened with
dlopen so nothing links frameworks).

- Build: TranslateCoreML is a lazy dependency, used only when the target is
  iOS / macOS (`-Dcoreml` turns it on elsewhere, for the reference checks).
  Windows and Linux builds never fetch or compile it; there `Backend.coreml`
  has a `noreturn` payload, so selecting it does not compile.
- `src/coreml/frontend.zig`: log-mel on the host CPU (max abs err vs the GPU
  1.7e-5 on hello.wav); frames that only see padding are skipped.
- `src/coreml/net.zig`: encoder (mel [1][80][3000] -> conv1/conv2 + GELU ->
  blocks -> cross K/V per decoder layer [1][H][1500][64]) and decoder step
  (x, mask, self cache sk/sv [1][H][448][64] in, kn/vn + logits out). fp16
  I/O, opset iOS 17 (CoreML7). Same math and tensor layout as the NPU graphs;
  LoRA merged into the decoder weights when the program is built.
- `src/coreml/engine.zig`: encode/step/pick/setAdapter like gpu/npu. Host:
  mel, embedding row, mask row, K/V row copy, greedy pick over fp16 logits.
  Inputs are MLMultiArrays over the engine's fp16 buffers (no copies); the
  encoder's outputs are read straight into the decoder's cross inputs.
  Programs are generated + compiled into `cache_dir` once (key: weights,
  adapter, net.zig source, opset, compute units). `runtime = .reference`
  runs the same programs on TranslateCoreML's host interpreter (any OS).
- Validated on the desktop (reference runtime, fp16 rounding emulated):
  tiny.en and base.en transcripts of hello.wav identical to the GPU's; cross K
  layer 0 max abs err 0.009 (tiny) / 0.03 (base) of max 6.5 / 7.8. `zig build
  test -Dcoreml` runs it (`src/coreml/tests.zig`, skipped without the model
  or GPU). Not run on Apple hardware yet: see docs/mac-session.md.
- `src/coreml/dev`: `zig build apple-check` (aarch64-ios and aarch64-macos
  objects of the library with the backend), `coreml-dev`
  (Mac or reference, compares with the GPU; `--export dir` writes the
  packages) and `SttCoreML.app` (`zig build ios` / `ios-sign` on the Mac).
- A Mac build needs no QAIRT headers: the NPU engine (the `qnn` module) is
  compiled only for aarch64-windows or with `-Dnpu`.

Next (Phase 2 continued): ANE-friendly layout ([1][C][1][S], conv 1x1,
per-head attention) if MLComputePlan shows CPU/GPU fallbacks; MLState (iOS 18)
for the self cache and cross K/V; LoRA as inputs (no recompile per adapter);
output backings.
