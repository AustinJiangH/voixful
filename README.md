# Voixful

A **SpeechAnalyzer-shaped** on-device speech API for Apple Silicon, with local ASR
backends running through **Core AI** (the WWDC 2026 framework that succeeds Core ML
for custom models). Mirrors Apple's `SpeechAnalyzer` / `SpeechTranscriber` so
developers can adopt it with near-mechanical renames — but runs your own model,
fully on device, on the ANE/GPU.

## What's here

Four ASR backends ship today, spanning four architectures:

- **Nemotron** — streaming RNN-T (`nvidia/nemotron-speech-streaming-en-0.6b`, vocab 1024).
- **Parakeet** — full-context TDT (`nvidia/parakeet-tdt-0.6b-v3` multilingual
  vocab 8192, or `-v2` English vocab 1024 — same pipeline, auto-detected from
  the asset's `pieces.txt`).
- **Cohere Transcribe** — attention encoder-decoder / AED (`CohereLabs/cohere-transcribe-03-2026`, vocab 16384).
- **Granite NAR** — non-autoregressive CTC + single-pass LLM transcript editor
  (`ibm-granite/granite-speech-4.1-2b-nar`, vocab 100352). Two dispatches per
  clip, no decode loop at all.

The Swift package layers them behind one model-agnostic façade:

- **`VoixfulSpeech`** — the brand core (model-agnostic): `VoixfulAnalyzer`, the
  `VoixfulSpeechModule` / `Transcribing` protocols, `AnalyzerInput`,
  `TranscriptionResult`. Depends on no model.
- **`VoixfulEngine`** — the engines themselves, organized per architecture under
  `Nemotron/`, `Parakeet/`, `Cohere/`, and `Shared/`.
- **`VoixfulNemotronStreaming`** / **`VoixfulParakeetTDT`** / **`VoixfulCohereTranscribe`**
  — thin per-architecture façade targets, each exposing a `Transcribing` conformer
  over its engine.
- **`VoixfulTUI`** — a reusable live-mic terminal UI (used by the CLI; works for
  any backend).
- **`VoixfulCLI`** — the `voixful` executable: `transcribe`, `mic`, `languages`.
- **`convert/`** — the Python pipelines that produce the Core AI `.aimodel` assets
  (upstream checkpoint → `torch.export` → `coreai_torch`), one dir per model, incl.
  **encoder compression**.
- A SwiftUI demo in `Demos/`.

## Quick start

```sh
swift build -c release
.build/release/voixful mic        --model path/to/nemotron-speech-streaming-en-0.6b-palette4.aimodel   # live mic
.build/release/voixful transcribe clip.wav --model path/to/nemotron-speech-streaming-en-0.6b-palette4.aimodel
.build/release/voixful languages
```

`voixful transcribe` **auto-detects** the backend from the `.aimodel`; override
with `--backend nemotron|parakeet|cohere` if needed.

The model `.aimodel` is **not** committed (it's gitignored, ~hundreds of MB to
GBs). Get it from Hugging Face (see [Distribution](#distribution)) or regenerate
it with `convert/` (see `convert/README.md`).

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
├── PLAN.md / HANDOFF.md        # design + state
├── Sources/
│   ├── VoixfulSpeech/          # brand core (analyzer + protocols + types)
│   ├── VoixfulTUI/             # reusable live-mic terminal UI
│   ├── VoixfulEngine/          # engines: Nemotron/ Parakeet/ Cohere/ Shared/
│   ├── VoixfulNemotronStreaming/   # NemotronStreamingTranscriber : Transcribing
│   ├── VoixfulParakeetTDT/         # ParakeetTDTTranscriber : Transcribing
│   ├── VoixfulCohereTranscribe/    # CohereTranscriber : Transcribing
│   └── VoixfulCLI/             # `voixful` executable
├── convert/                    # model conversion, one dir per model
│   ├── nemotron-streaming/     # NeMo → coreai_torch → .aimodel (+ --quantize)
│   ├── parakeet-tdt/           # full-context TDT pipeline
│   └── cohere-transcribe/      # AED pipeline
├── Tests/ · scripts/ · Demos/ · fixtures/
```

## Distribution

- **Code → GitHub** (this repo).
- **Model weights → Hugging Face** — `.aimodel` files are large binaries; HF is
  built for them (LFS, model cards, versioning). GitHub is poor for >100 MB blobs.
  Publish the `.aimodel` + `pieces.txt` + `MODEL_CARD.md` to an HF model repo and
  have the convert pipeline / a download step fetch it.

## License (dual)

Two licenses, by artifact type:

- **Code in this repo → Apache License 2.0** — see [`LICENSE`](LICENSE) and
  [`NOTICE`](NOTICE).
- **Model weights** (the `.aimodel` artifacts and any quantized/palettized
  derivatives) → **[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)**,
  as derivatives of
  [`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
  (permits commercial use, redistribution, and derivatives — retain attribution +
  the license). See [`MODEL_CARD.md`](MODEL_CARD.md).

The weights are **not** Apache-licensed; keep the two clearly separated when you
redistribute.
