# Voixful

A **SpeechAnalyzer-shaped** on-device speech API for Apple Silicon, with a local
Nemotron 3.5 ASR backend running through **Core AI** (the WWDC 2026 framework that
succeeds Core ML for custom models). Mirrors Apple's `SpeechAnalyzer` /
`SpeechTranscriber` so developers can adopt it with near-mechanical renames — but
runs your own model, fully on device, on the ANE/GPU.

## What's here

- **`VoixfulSpeech`** — the brand core (model-agnostic): `VoixfulAnalyzer`, the
  `VoixfulSpeechModule` / `Transcribing` protocols, `AnalyzerInput`,
  `TranscriptionResult`. Depends on no model.
- **`VoixfulNemotronStreaming`** — the Nemotron backend: `NemotronStreamingTranscriber: Transcribing`
  wrapping the streaming RNN-T engine.
- **`VoixfulTUI`** — a reusable live-mic terminal UI (used by the CLI; works for
  any backend).
- **`voixful`** CLI — `transcribe`, `mic`, `languages`.
- **`convert/`** — the Python pipeline that produces the Core AI `.aimodel`
  (NeMo → `torch.export` → `coreai_torch`), incl. **encoder compression**.
- Legacy `NemotronASR` library + `nemotron-asr` CLI (kept during migration) and a
  SwiftUI demo in `Demos/`.

## Quick start

```sh
swift build -c release
.build/release/voixful mic        --model path/to/model.aimodel        # live mic
.build/release/voixful transcribe clip.wav --model path/to/model.aimodel --streaming
.build/release/voixful languages
```

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
│   ├── VoixfulNemotronStreaming/        # NemotronStreamingTranscriber : Transcribing
│   ├── VoixfulCLI/             # `voixful` executable
│   ├── NemotronASR/            # streaming RNN-T engine (mel, sessions, decoder)
│   └── NemotronASRCLI/         # legacy `nemotron-asr` executable
├── convert/                    # model conversion, one dir per model
│   └── nemotron/               # NeMo → coreai_torch → .aimodel (+ --quantize)
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
