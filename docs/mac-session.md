# Mac session: verify the Core ML backend on Apple hardware

Everything below was built and checked on Windows except running Core ML
itself: `src/coreml/dev` cross-compiles stt with the backend for
aarch64-ios / aarch64-macos and cross-links `coreml-dev` for aarch64-macos,
and the reference interpreter reproduces the GPU's transcripts
(`zig build test -Dcoreml`). This session checks the real Core ML runtime on
a Mac and on the iPhone. Work through it top to bottom and keep the output of
every step marked **record**.

Background: [NOTES.md, "iPhone: Core ML backend"](../NOTES.md#iphone-core-ml-backend-2026-11-12)
and the step list at the top of [src/coreml/dev/build.zig](../src/coreml/dev/build.zig).

## 0. Setup

1. Apple-silicon Mac with Xcode (iPhoneOS SDK, `xcrun`, `devicectl`), and
   Zig 0.18.0-dev.1005 active (`zig version`).
2. This repo on `main`, plus a checkout of
   [TranslateCoreML](https://github.com/Mauhlt/TranslateCoreML) for step 1.
   stt fetches its dependencies itself (`Audio`, and `TranslateCoreML` lazily
   for Apple targets; pinned in `build.zig.zon`), so the pinned commits must
   be on GitHub. No QAIRT SDK or QNN headers are needed: the NPU engine is
   compiled only for aarch64-windows.
3. `models/ggml-tiny.en.bin` in this repo ([README, Setup](../README.md#setup)
   shows the download; use the tiny.en file). `testdata/hello.wav` is in the repo.
4. For the iPhone (step 4): a development signing identity, a provisioning
   profile that lists the iPhone, the team id, and the iPhone connected with
   Developer Mode on.

## 1. TranslateCoreML probe on the Mac (Core ML and the Neural Engine at all)

In the TranslateCoreML checkout: `zig build test`, then the steps in its
README, [On a Mac (no iPhone needed)](https://github.com/Mauhlt/TranslateCoreML#on-a-mac-no-iphone-needed).

Success: `zig build test` passes; every probe model reports `rel err` below
1e-2 (OK) for all four compute-unit settings; the `plan` lines show ops on
`neural_engine` for `cpu_and_neural_engine`; and `block` has a clearly lower
`median_ms` with `cpu_and_neural_engine` than with `cpu_only`.
**Record** `probe-mac.json`.

## 2. stt builds on the Mac

From this repo:

```sh
zig build test            # macOS target: the Core ML backend is on by default
cd src/coreml/dev
zig build apple-check ios-check
zig build                 # zig-out/bin/coreml-dev for this Mac
cd ../../..
```

Success: all three finish without errors. In `zig build test` the tests that
need the NVIDIA GPU report as skipped. If `zig build test` fails outside
`src/coreml` (the desktop CLI or stt-local for macOS), **record** it and go
on: only the library and `src/coreml/dev` matter for the iPhone.

## 3. coreml-dev on the Mac (the real stt programs on Core ML)

From this repo (the default model and audio paths are relative to it):

```sh
D=src/coreml/dev/zig-out/bin/coreml-dev
$D --runs 3                    # Neural Engine (cpu_and_neural_engine), first run generates + compiles
$D --runs 3                    # again: now "(cache)"
$D --units gpu
$D --units cpu
$D --units all --adapter
$D --runtime reference         # host interpreter on the Mac, for comparison
```

There is no NVIDIA GPU on the Mac, so each run first prints
`GPU reference unavailable: ...`; that is expected.

Success, for every line above:

- `Core ML graphs ... ready in N ms`, and on the second run both programs
  say `(cache)`.
- `Core ML: "Hello World. This is a test of speech recognition written in Piersig."`,
  the text the GPU and the reference interpreter give for tiny.en. A
  one-word difference on a Core ML run is fp16 rounding and worth noting,
  not a failure; empty or garbled text is a failure.
- With `--adapter`: `setAdapter: ...`, then `base again: ... (same as before)`.
- `run 2` and `run 3` are faster than `run 1`. Note how the ANE run's
  encoder ms and `ms/step` compare with `--units gpu` and `--units cpu`.

**Record** the whole output of each run (the timings matter for the iPhone
plan).

## 4. SttCoreML.app on the iPhone

Signing variables and `devicectl` usage are as in TranslateCoreML's README,
[On the iPhone](https://github.com/Mauhlt/TranslateCoreML#on-the-iphone);
only the build step, app path and bundle id differ:

```sh
cd src/coreml/dev
zig build ios-sign -Dios-bundle-id=<bundle id of the profile>
xcrun devicectl list devices
xcrun devicectl device install app --device <UDID> zig-out/ios/SttCoreML.app
xcrun devicectl device process launch --console --device <UDID> <bundle id>
```

The app bundles the tiny.en model and `hello.wav` (`-Dios-model=<path>` for
another ggml file). It shows its log on screen, NSLogs it, and writes
`Documents/stt-coreml.txt`; fetch it with `devicectl device copy from`
exactly as in that README, with `--source Documents/stt-coreml.txt`.

Success:

- `devices: cpu=true gpu=true neural_engine=true (ANE cores N)`.
- `[cpu_and_neural_engine] ready in ... (generated + compiled)` on the first
  launch and `(cache)` for both programs after relaunching the app.
- Three `[cpu_and_neural_engine] run` lines with the transcript from step 3,
  then the same for `[cpu_and_gpu]` and `[cpu_only]` with no
  `(differs from the Neural Engine transcript)` line.
- `[ane] setAdapter ...` and a `with adapter:` transcript.
- The last line is `done`.

Failure looks like `FAILED: <error> (<Core ML message>)` or
`[<units>] init failed: ...`. **Record** `stt-coreml.txt` from the first and
the second launch.

## 5. What to send back

`probe-mac.json`, the step 2 result, the step 3 outputs, and both
`stt-coreml.txt` files. They decide whether the next step is wiring
`stt.Transcriber` with `.coreml` into Portal's iPhone app or reworking the
graph layout for the Neural Engine (NOTES.md, "Next").
