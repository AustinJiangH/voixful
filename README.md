# Voixful

A **SpeechAnalyzer-shaped** on-device speech API for Apple Silicon, with local ASR
backends running through **Core AI** (the WWDC 2026 framework that succeeds Core ML
for custom models). Mirrors Apple's `SpeechAnalyzer` / `SpeechTranscriber` so
developers can adopt it with near-mechanical renames — but runs your own model,
fully on device, on the ANE/GPU.

## What's here

Three ASR backends ship today, spanning three architectures:

- **Nemotron** — streaming RNN-T (`nvidia/nemotron-speech-streaming-en-0.6b`, vocab 1024).
- **Parakeet** — full-context TDT (`nvidia/parakeet-tdt-0.6b-v3`, vocab 8192).
- **Cohere Transcribe** — attention encoder-decoder / AED (`CohereLabs/cohere-transcribe-03-2026`, vocab 16384).

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
