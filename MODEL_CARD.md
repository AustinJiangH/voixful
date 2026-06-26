---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/nemotron-speech-streaming-en-0.6b
language:
  - en
pipeline_tag: automatic-speech-recognition
tags:
  - core-ai
  - aimodel
  - apple-silicon
  - streaming-asr
  - rnnt
  - palettization
  - quantization
---

# Nemotron Speech Streaming 0.6B — Core AI (`.aimodel`), 4-bit palettized

A streaming ASR model for **Apple Silicon**, converted from
[`nvidia/nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b)
to Apple's **Core AI** `.aimodel` format and **4-bit k-means palettized** so the
Neural Engine keeps the weights compressed at runtime. Used by the
<!-- TODO: replace placeholder repo URL with the real one before publishing -->
[Voixful](https://github.com/your-org/voixful) Swift package.

## Attribution & License

Derived from NVIDIA's `nemotron-speech-streaming-en-0.6b`. Use of these weights —
including this converted/palettized derivative — is governed by the
[**NVIDIA Open Model License Agreement**](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
which permits commercial use, redistribution, and derivatives. Retain this
attribution and a copy of the license when redistributing.

The conversion was performed with Apple's `coreai-torch` / `coreai-opt`; those
tools place no license claim on the output.

## Variants

Only two variants ship: FP16 and 4-bit palettized (no standalone INT8 asset).

| File | Format | Disk | Runtime RAM | Speed (5.5 s clip) |
|---|---|---|---|---|
| `nemotron-speech-streaming-en-0.6b-fp16.aimodel` | FP16 | 2.3 GB | 4.63 GB | 0.62 s |
| **`nemotron-speech-streaming-en-0.6b-palette4.aimodel`** | **4-bit palettization** | **328 MB** | **0.66 GB** | **0.37 s** |

Measured on **M5 Pro / macOS 27**, streaming path, same `sample.wav`. Both
transcribe correctly. **Palettization is the recommended variant** — it's the
format Apple's Neural Engine keeps compressed in memory (just-in-time
decompression per layer), so it cuts **disk + RAM + latency**, not just disk.

## Contents

Each `.aimodel` is a self-contained directory: `main.mlirb` + `main.hash` +
`metadata.json`, containing three named functions — `encode`, `predict_step`,
`join` — plus the SentencePiece vocab `pieces.txt` bundled inside (no loose
sibling vocab file).

> Note: these conversions currently lack a baked `preprocess_audio` function, so
> they run via the **streaming** path (the Voixful `feed()`/`mic` path). A batch
> file-transcribe path needs the preprocessor baked in (see the repo's
> `convert/` pipeline).

## Usage (Voixful, Swift)

```sh
voixful mic --model path/to/nemotron-speech-streaming-en-0.6b-palette4.aimodel
voixful transcribe clip.wav --model path/to/nemotron-speech-streaming-en-0.6b-palette4.aimodel
```

## How it was made

```sh
# in convert/nemotron-streaming, with the coreai toolchain (pydantic<2.10):
python -m nemotron_convert.convert --quantize palette4 \
       --output nemotron-speech-streaming-en-0.6b-palette4.aimodel
```

See the repo's `convert/README.md` and the design notes for the full recipe.

## Limitations

- English only (the multilingual sibling needs an unreleased NeMo class).
- 4-bit palettization holds accuracy on our sample; **validate WER on a real
  eval set** before production. A 6-bit variant (`--quantize palette6`) is a
  safer fallback.
- Built with pre-release Apple Core AI tooling (`coreai-torch 0.4.0`,
  `coreai-opt 0.2.0`); re-verify on newer releases.
