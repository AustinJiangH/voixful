# convert/ — model conversion pipelines (one directory per model)

Python tooling that converts upstream ASR checkpoints into the Core AI
`.aimodel` assets the Swift runtime loads. Organized **model by model**, mirroring
the `Sources/` backend layout (`VoixfulNemotron`, future `VoixfulParakeet`, …):
each model's conversion is self-contained — its own `pyproject.toml` (deps differ
per model — Nemotron needs NeMo), its own package, its own README.

```
convert/
├── README.md                 ← you are here (brand overview)
├── nemotron/                 ← Nemotron 3.5 ASR Streaming 0.6B (RNN-T, streaming)
│   ├── pyproject.toml · README.md
│   └── nemotron_convert/     ← convert.py / extract.py / export.py / models.py / …
├── parakeet/                 ← Parakeet TDT 0.6B (TDT, full-context)
│   ├── pyproject.toml · README.md
│   └── parakeet_convert/     ← convert.py / extract.py / export.py / models.py
└── cohere-transcribe/                   ← Cohere Transcribe 2B (attention encoder-decoder)
    ├── pyproject.toml · README.md   ← gated model — authenticate first (see README)
    └── cohere_convert/       ← convert.py / extract.py / export.py / models.py
```

Each model is self-contained (own `pyproject.toml`, own package, own license note);
both expose `convert.py --quantize {none,int8,palette4,palette6}`.

## Adding a new model

Create a sibling directory `convert/<model>/` with its own `pyproject.toml` and a
`<model>_convert/` package exposing a `convert.py` that emits a `.aimodel`. The
Swift side adds a matching `Voixful<Model>` backend conforming to `Transcribing`.
Nothing here is shared across models, so a new model's heavy/odd deps never leak
into the others.

## Encoder compression — `--quantize`

`convert.py` can compress the encoder via `coreai_opt`. Requires the `presets`
API and **`pydantic<2.10`** (the `>=2.0` default pulls a 2.14 alpha that breaks
coreai_opt's config models).

- `int8` / `int8-mmap` — INT8 **linear** weight-only (`QuantizerConfig.presets.w8()`).
- `palette4` / `palette6` — 4/6-bit **k-means palettization** (`KMeansPalettizerConfig.presets.w4/.w6`).

**Measured on M5 Pro (5.5 s `sample.wav`, streaming), vs the FP16 baseline:**

| Mode | `.aimodel` size | Runtime RAM | Speed | Transcript |
|---|---|---|---|---|
| FP16 (default) | 2.3 GB | 4.63 GB | 0.62 s | ✅ |
| `int8` (linear) | 619 MB (3.7×) | 4.63 GB | 0.67 s | ✅ |
| **`palette4` (4-bit)** | **328 MB (7×)** | **0.66 GB (7×)** | **0.37 s (1.7×)** | ✅ |

**Key finding: the compression *type* is everything.**
- **Linear INT8** cuts disk only — runtime RAM is unchanged, because Core AI
  specializes the `.aimodel` to float weights at load (W8A16 dequantizes for the
  FP16 matmuls). Storage/bandwidth win, not a memory win.
- **Palettization** keeps weights low-bit *in memory* and decompresses them
  just-in-time per layer, so we get **7× less RAM and 1.7× faster**, not just
  smaller disk. This is the lever to use for a memory-constrained deployment.

So for compression, palettization — not linear int8 — is the format that wins at
runtime, not only on disk.

Caveat: this convert pipeline predates the out-of-repo `preprocess_audio` baking,
so these compressed models run only via the streaming `feed()` path (`--streaming`)
— the batch `transcribe()` path needs the baked preprocessor. The existing Swift
runtime loads + runs the palettized `.aimodel` as-is (no separate code needed).

## License

- This **conversion code** → Apache License 2.0 (repo root `LICENSE` / `NOTICE`).
- The **`.aimodel` it produces** (and any quantized/palettized derivative) →
  derived from `nvidia/nemotron-speech-streaming-en-0.6b`, governed by the
  [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
  Retain attribution + the license when redistributing. See `../MODEL_CARD.md`.
