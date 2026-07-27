# Voixful

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Platform: macOS 27+ · Apple Silicon](https://img.shields.io/badge/platform-macOS%2027%2B%20·%20Apple%20Silicon-lightgrey.svg)
![Swift 6](https://img.shields.io/badge/Swift-6-orange.svg)

A **SpeechAnalyzer-shaped** on-device speech API for Apple Silicon, with local ASR
backends running through **Core AI** (the WWDC 2026 framework that succeeds Core ML
for custom models). Mirrors Apple's `SpeechAnalyzer` / `SpeechTranscriber` so
developers can adopt it with near-mechanical renames — but runs your own model,
fully on device, on the ANE/GPU.

## What's here

Model checkpoints ship today across seven backends, spanning five
architecture families (Parakeet v2 and v3 share the `VoixfulParakeetTDT` backend;
Cohere and Whisper are both attention encoder-decoders; Canary-Qwen and Qwen3-ASR
are both SALM). The backend is auto-detected from the `.aimodel`'s function
signatures — same `transcribe` / `mic` commands for all of them.

| Model | Params | Architecture | Native streaming | Live mic | Languages | WER¹ |
|---|---|---|---|---|---|---|
| **Nemotron** `nvidia/nemotron-speech-streaming-en-0.6b` | 0.6 B | cache-aware streaming RNN-T | **yes** — partials every ~0.7 s while you speak | best latency | English only | 5.73 |
| **Parakeet v2** `nvidia/parakeet-tdt-0.6b-v2` | 0.6 B | full-context TDT | no (re-transcribe partials) | great (fastest passes, ~70 ms) | English only | 5.39 |
| **Parakeet v3** `nvidia/parakeet-tdt-0.6b-v3` | 0.6 B | full-context TDT | no (re-transcribe partials) | great (~75 ms passes) | **25 languages** (European incl. en/de/fr/es/it/pt/pl/uk/ru…) | 5.66 |
| **Granite NAR** `ibm-granite/granite-speech-4.1-2b-nar` | 2.2 B | CTC + single-pass LLM editor (no decode loop) | no (re-transcribe partials) | good (~0.3 s passes) | **en, fr, de, es, pt** | 4.95 (best) |
| **Cohere Transcribe** `CohereLabs/cohere-transcribe-03-2026` | 2 B | attention encoder-decoder | no (re-transcribe partials) | ok (~0.3 s passes) | **14 languages** | 5.20 |
| **Canary-Qwen** `nvidia/canary-qwen-2.5b` | 2.5 B | SALM: Conformer + Qwen3-1.7B | no (re-transcribe partials) | sluggish (~3–4 s passes) — prefer for files | English only | 5.06 |
| **Whisper** `openai/whisper-base` (…tiny/small/large-v3) | 74 M–1.5 B | attention encoder-decoder (fixed 30 s window) | no (re-transcribe partials) | good (30 s single-pass) | **multilingual** (99 languages) | 7.5² |
| **Qwen3-ASR** `Qwen/Qwen3-ASR-0.6B` / `-1.7B` | 0.6 B / 1.7 B | SALM: audio encoder + Qwen3 decoder | no (re-transcribe partials) | files (provisional) | **multilingual** (52 languages) | 5.76³ |

¹ Open ASR Leaderboard mean WER (lower is better), for the source checkpoints.
² Whisper WER is size-dependent (base ≈ 7–10%, large-v3 ≈ 4%); the row lists the `-base` default.

³ Qwen3-ASR WER is the reported Qwen3-ASR-1.7B multilingual average; the 0.6B variant is not yet on the leaderboard. The **0.6B is converted and validated on-device** (`convert/qwen3-asr`, palette4 encoder + palette6 decoder; transcribes the fixture exactly). It's not yet published for download, so import the locally-converted asset to run it. Compute placement: the engine specializes the asset twice and defaults to **encode→CPU, decode→ANE** (~2.75 s / 5.5 s clip) — ANE fp16 flips argmax in the audio encoder, so the encoder stays on CPU. The 1.7B is registered but not yet converted.

**Streaming vs. live mic.** Only Nemotron streams natively (its encoder is
cache-aware, so words appear *while* you speak). Every other backend is
full-context: `mic` still works — the live wrapper re-transcribes the growing
utterance and commits chunks at silence gaps — but the refresh cadence is the
backend's full-pass latency. Practically: Nemotron for lowest-latency dictation,
Granite for the most accurate live+file balance, Parakeet v2 for the smallest
footprint, Canary-Qwen when you want punctuated + capitalized file transcripts.

The Swift package layers them behind one model-agnostic façade:

- **`VoixfulSpeech`** — the brand core (model-agnostic): `VoixfulAnalyzer`, the
  `VoixfulSpeechModule` / `Transcribing` protocols, `AnalyzerInput`,
  `TranscriptionResult`. Depends on no model.
- **`VoixfulEngine`** — the engines themselves, organized per architecture under
  `Nemotron/`, `Parakeet/`, `Cohere/`, `Granite/`, `Canary/`, `Whisper/`, and `Shared/`.
- **`VoixfulNemotronStreaming`** / **`VoixfulParakeetTDT`** / **`VoixfulCohereTranscribe`**
  / **`VoixfulGraniteNAR`** / **`VoixfulCanaryQwen`** / **`VoixfulWhisper`** — thin
  per-architecture façade targets, each exposing a `Transcribing` conformer over its engine.
- **`VoixfulFullContext`** — the live-mic wrapper for the non-streaming backends
  (`LiveFullContextTranscriber`): re-transcribes the growing utterance on a
  throttle and commits at silence gaps, giving Parakeet/Cohere/Granite/Canary a
  streaming-like `mic` experience.
- **`VoixfulTUI`** — a reusable live-mic terminal UI (used by the CLI; works for
  any backend).
- **`VoixfulCLI`** — the `voixful` executable: `transcribe`, `mic`, `languages`.
- **`convert/`** — the Python pipelines that produce the Core AI `.aimodel` assets
  (upstream checkpoint → `torch.export` → `coreai_torch`), one dir per model, incl.
  **encoder compression**.

## Requirements

- **Apple Silicon Mac** (M-series) — the ANE/GPU paths assume it.
- **macOS 27** or later — Core AI is the WWDC 2026 runtime (`.macOS("27.0")` in
  `Package.swift`).
- **Swift 6 toolchain** (Xcode 27 / the matching command-line tools).
- A converted **`.aimodel`** for at least one backend — download it from Hugging
  Face (see [Distribution](#distribution)) or build it with `convert/`.

Core AI is pre-release: assets must be exported with **coreai-torch ≥ 0.4.1**
(0.4.0-era exports fail to load from macOS 27 beta 3 on).

## Quick start

```sh
swift build -c release
.build/release/voixful mic        --model path/to/nemotron-speech-streaming-en-0.6b-palette4.aimodel   # live mic
.build/release/voixful transcribe clip.wav --model path/to/nemotron-speech-streaming-en-0.6b-palette4.aimodel
.build/release/voixful languages
```

`voixful transcribe` **auto-detects** the backend from the `.aimodel`; override
with `--backend nemotron|parakeet|cohere|granite|canary|whisper|qwen3` if needed.

The model `.aimodel` is **not** committed (it's gitignored, ~hundreds of MB to
GBs). Get it from Hugging Face (see [Distribution](#distribution)) or regenerate
it with `convert/` (see `convert/README.md`).

## Using the Swift API

Add the package as a dependency and import the brand core plus the backend
target(s) you want:

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/AustinJiangH/voixful.git", from: "0.1.0"),
],
targets: [
    .target(name: "MyApp", dependencies: [
        .product(name: "VoixfulSpeech", package: "voixful"),
        .product(name: "VoixfulNemotronStreaming", package: "voixful"),
    ]),
]
```

The surface mirrors Apple's `SpeechAnalyzer` / `SpeechTranscriber`: build a
`Transcribing` module, hand it to a `VoixfulAnalyzer`, feed `AnalyzerInput`
buffers, and read `TranscriptionResult`s off the module's `results` stream.

```swift
import VoixfulSpeech
import VoixfulNemotronStreaming

let transcriber = try NemotronStreamingTranscriber(
    modelURL: URL(fileURLWithPath: "nemotron-...-palette4.aimodel"),
    locale: Locale(identifier: "en-US"),
    mode: .streaming,
    reportingOptions: [.volatileResults],   // emit partials while you speak
    attributeOptions: [.audioTimeRange]
)
let analyzer = VoixfulAnalyzer(modules: [transcriber])

// Read results concurrently with feeding audio.
let reader = Task {
    for try await result in transcriber.results where result.isFinal {
        print(String(result.text.characters))
    }
}

let (inputs, input) = AsyncStream<AnalyzerInput>.makeStream()
try await analyzer.start(inputSequence: inputs)
input.yield(AnalyzerInput(buffer: pcmBuffer, bufferStartTime: .zero))
input.finish()
try await analyzer.finalizeAndFinishThroughEndOfInput()
_ = try await reader.value
```

Every backend conforms to the same `Transcribing` protocol, so switching models
is a near-mechanical rename — swap `NemotronStreamingTranscriber` for
`ParakeetTDTTranscriber`, `CohereTranscriber`, `GraniteTranscriber`,
`CanaryTranscriber`, or `WhisperTranscriber` (each in its own product). Core types live in
`VoixfulSpeech`: `VoixfulAnalyzer`, the `VoixfulSpeechModule` / `Transcribing`
protocols, `AnalyzerInput`, `TranscriptionResult`, `ReportingOption`,
`ResultAttributeOption`. `VoixfulCLI/main.swift` is a full worked example.

## Building & testing

```sh
swift build -c release          # build the package + the `voixful` CLI
swift test                      # unit tests (fixture-gated tests XCTSkip when
                                # the .aimodel / audio fixtures are absent — see
                                # fixtures/README.md, so a clean checkout passes)
scripts/verify-backends.sh      # end-to-end smoke across the installed backends
```

## Models & compression

The encoder can be compressed via `convert.py --quantize`. Measured on M5 Pro
(`sample.wav`, streaming):

| Variant | Disk | Runtime RAM | Speed | Transcript |
|---|---|---|---|---|
| FP16 | 2.3 GB | 4.63 GB | 0.62 s | ✅ |
| INT8 (linear) | 619 MB | 4.63 GB | 0.67 s | ✅ |
| **4-bit palettization** | **328 MB** | **0.66 GB** | **0.37 s** | ✅ |

**Palettization is the win** — the Neural Engine keeps it compressed in memory
(just-in-time decompression), so it cuts **disk + RAM + latency**, not just disk.
Linear INT8 cuts disk only. See `convert/README.md` for the recipe and `MODEL_CARD.md`.

## Performance knobs & measured baselines

Assets must be exported with **coreai-torch ≥ 0.4.1** — 0.4.0-era exports fail
to load from macOS 27 beta 3 on ("Failed to convert to versioned IR").

Warm per-utterance numbers, M5 Pro, 5.5 s clip, palette4 assets:

| Backend | Wall (session-style) | Stage split | Notes |
|---|---|---|---|
| Nemotron (streaming) | ~0.15 s | mel 1 / encode 138 / decode ~7 ms | partials every 0.72 s with `--chunk 560` (was 1.28 s) |
| Nemotron (batch) | ~0.18 s | mel 7 / encode 160 / decode 14 ms | better transcripts (full-context mel); needs `preprocess_audio` in the asset |
| Parakeet (full-context) | ~0.07 s | mel 2 / encode 56 / decode 15 ms | CPU transducer decode via the `.transducer` sidecar |
| Cohere 2B (AED) | ~0.29 s | mel 1 / encode 127 / decode 160 ms | + ~1.1 s one-time encoder window load, hidden by `prepare()` prewarm in session use |
| Parakeet v2 (English) | ~0.07 s | mel 2 / encode 58 / decode 11 ms | best English WER of the small tier (5.39 leaderboard avg) |
| Granite NAR 2B | ~0.29 s | encode 115 / edit 170 ms (mel in-graph) | best WER we ship (4.95); NO autoregressive loop — two dispatches per clip; `GRANITE_TRACE=1` |
| Canary-Qwen 2.5B | ~4 s | encode ~1-2 s / decode ~150 ms/token | best punctuation+casing; decode is state-transfer-bound (235 MB KV round-trip/step) — x4 unrolling + fp16 states are the known fixes; `CANARY_TRACE=1` |

- **Multi-window encoders** (all three backends): the runtime picks the smallest
  exported `encode_<frames>` that fits the clip (Cohere ladder 220–3500, Parakeet
  125–1000, Nemotron chunk tiers 32/72/128/256 — small Nemotron windows are traced
  under their trained att-context tier).
- **Prewarm**: `prepare()` background-loads likely inference functions so weight
  materialization overlaps audio capture (`VOIXFUL_PREWARM=0` disables;
  `VOIXFUL_SETTLE_MS=<ms>` emulates capture idle for CLI benchmarking).
- **Tracing**: `NEMOTRON_TRACE=1` / `PARAKEET_TRACE=1` / `COHERE_TRACE=1` print
  per-utterance stage breakdowns to stderr.
- **Placement**: `VOIXFUL_COMPUTE=gpu|ane|cpu` pins whole-model specialization;
  `VOIXFUL_COMPUTE_DECODE=...` pins the decode-side second instance;
  `VOIXFUL_COHERE_DECODER_ASSET=<path>` sources Cohere decoder functions from the
  decoder-only companion asset. Measured caveat: on M5-class silicon the
  autoregressive decode floor (~4.6 ms/step Cohere, ~0.5 ms/dispatch transducer)
  is **placement-, kernel-count-, weight-bandwidth-, and payload-invariant** —
  default auto-partition is already optimal; don't chase pins without new evidence.

## Repo layout

```
voixful/
├── Package.swift
├── LICENSE                     # Apache-2.0 (code only)
├── NOTICE                      # attributions (incl. the model-weights split)
├── MODEL_CARD.md               # for the Hugging Face model upload
├── Sources/
│   ├── VoixfulSpeech/          # brand core (analyzer + protocols + types)
│   ├── VoixfulTUI/             # reusable live-mic terminal UI
│   ├── VoixfulEngine/          # engines: Nemotron/ Parakeet/ Cohere/ Granite/ Canary/ Whisper/ Qwen3ASR/ Shared/
│   ├── VoixfulNemotronStreaming/   # NemotronStreamingTranscriber : Transcribing
│   ├── VoixfulParakeetTDT/         # ParakeetTDTTranscriber : Transcribing
│   ├── VoixfulCohereTranscribe/    # CohereTranscriber : Transcribing
│   ├── VoixfulGraniteNAR/          # GraniteTranscriber : Transcribing
│   ├── VoixfulCanaryQwen/          # CanaryTranscriber : Transcribing
│   ├── VoixfulWhisper/             # WhisperTranscriber : Transcribing
│   ├── VoixfulQwen3ASR/            # Qwen3ASRTranscriber : Transcribing
│   ├── VoixfulFullContext/         # LiveFullContextTranscriber (live mic for the above)
│   └── VoixfulCLI/             # `voixful` executable
├── convert/                    # model conversion, one dir per model
│   ├── nemotron-streaming/     # NeMo → coreai_torch → .aimodel (+ --quantize)
│   ├── parakeet-tdt/           # full-context TDT pipeline
│   ├── cohere-transcribe/      # attention encoder-decoder pipeline
│   ├── granite-speech/         # CTC + single-pass LLM editor (NAR) pipeline
│   ├── canary-qwen/            # SALM (Conformer + Qwen3) pipeline
│   ├── qwen3-asr/              # SALM (audio encoder + Qwen3) pipeline — scaffold
│   └── whisper/                # attention encoder-decoder pipeline (fixed 30 s)
├── Tests/ · scripts/ · fixtures/
```

## Distribution

- **Code → GitHub** (this repo).
- **Model weights → Hugging Face** — `.aimodel` files are large binaries; HF is
  built for them (LFS, model cards, versioning). GitHub is poor for >100 MB blobs.
  Publish the `.aimodel` (with `pieces.txt` bundled inside it) + `MODEL_CARD.md`
  to an HF model repo and have the convert pipeline / a download step fetch it.

## Contributing

Issues and pull requests are welcome. A good first contribution is **a new
backend**: add a `convert/<model>/` pipeline that emits a `.aimodel`, then a
matching `Voixful<Model>` target conforming to `Transcribing` — the analyzer only
knows the protocol, so nothing else changes. See [`CONTRIBUTING.md`](CONTRIBUTING.md)
for the build/test loop and the backend checklist.

## License (dual)

Two licenses, by artifact type:

- **Code in this repo → Apache License 2.0** — see [`LICENSE`](LICENSE) and
  [`NOTICE`](NOTICE).
- **Model weights** (each `.aimodel` and any quantized/palettized derivative) →
  **the license of the upstream checkpoint it derives from**, which differs per
  model. Retain the upstream attribution + license when you redistribute:

  | Backend | Upstream checkpoint | Weights license |
  |---|---|---|
  | Nemotron | `nvidia/nemotron-speech-streaming-en-0.6b` | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) |
  | Parakeet v2/v3 | `nvidia/parakeet-tdt-0.6b-v2` / `-v3` | CC-BY-4.0 |
  | Cohere Transcribe | `CohereLabs/cohere-transcribe-03-2026` | upstream (gated — see `convert/cohere-transcribe/README.md`) |
  | Granite NAR | `ibm-granite/granite-speech-4.1-2b-nar` | Apache-2.0 |
  | Canary-Qwen | `nvidia/canary-qwen-2.5b` | CC-BY-4.0 |
  | Whisper | `openai/whisper-*` | MIT |
  | Qwen3-ASR | `Qwen/Qwen3-ASR-0.6B` / `-1.7B` | Apache-2.0 |

  See [`MODEL_CARD.md`](MODEL_CARD.md) and the per-model `convert/*/README.md`.

The weights are **not** covered by this repo's Apache license; keep the two
clearly separated when you redistribute.
