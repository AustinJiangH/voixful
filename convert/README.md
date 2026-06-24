# convert/ — model conversion pipelines (one directory per model)

Python tooling that converts upstream ASR checkpoints into the Core AI
`.aimodel` assets the Swift runtime loads. Organized **model by model**, mirroring
the `Sources/` backend layout (`VoixfulNemotron`, future `VoixfulParakeet`, …):
each model's conversion is self-contained — its own `pyproject.toml` (deps differ
per model — Nemotron needs NeMo), its own package, its own README.

```
convert/
├── README.md                 ← you are here (brand overview)
└── nemotron/                 ← Nemotron 3.5 ASR Streaming 0.6B
    ├── pyproject.toml
    ├── README.md             ← install + run instructions
    └── nemotron_convert/     ← NeMo → torch.export → coreai_torch
        ├── convert.py        ← entry point (emits nemotron-streaming.aimodel)
        ├── extract.py        ← pull nn.Modules out of the .nemo checkpoint
        ├── export.py         ← torch.export each (encode / predict_step / join)
        ├── models.py         ← Core-AI-shaped wrappers
        ├── verify_parity.py  ← FP32 vs converted SNR check
        └── smoke.py / *_demo.py
```

## Adding a new model

Create a sibling directory `convert/<model>/` with its own `pyproject.toml` and a
`<model>_convert/` package exposing a `convert.py` that emits a `.aimodel`. The
Swift side adds a matching `Voixful<Model>` backend conforming to `Transcribing`.
Nothing here is shared across models, so a new model's heavy/odd deps never leak
into the others.

## Quantization (INT8 encoder)

`nemotron/.../convert.py --quantize int8` applies `coreai_opt`'s INT8 weight-only
preset (`QuantizerConfig.presets.w8()`) to the encoder. Requires `coreai_opt` with
the `presets` API and **`pydantic<2.10`** (the `>=2.0` default pulls a 2.14 alpha
that breaks coreai_opt's config models).

**Measured on M5 Pro (5.5 s `sample.wav`), vs the FP16 baseline:**

| Mode | `.aimodel` size | Peak RAM | Transcript |
|---|---|---|---|
| FP16 (default) | 2.3 GB | ~5.0 GB | ✅ |
| `--quantize int8` (graph) | **619 MB (3.7×)** | ~5.0 GB | ✅ |
| `--quantize int8-mmap` (eager) | 619 MB (+584 MB redundant) | ~5.0 GB | ✅ |

**Key finding: quantization cuts on-disk size ~3.7× (matching FluidAudio's Int8
~584 MB) with no accuracy loss — but does NOT reduce runtime RAM.** Core AI
specializes the `.aimodel` to float weights at load, so RSS stays ~5 GB regardless
of int8 storage; the `int8-mmap` path doesn't help because `coreai_torch` bakes the
weights into the self-contained `.aimodel` (the mmap safetensors are redundant).
FluidAudio's ~80 MB RSS comes from CoreML mmap'ing int8 weights at runtime — not
reproducible through the Core AI `.aimodel` + Swift runtime as it stands.

Caveat: this convert pipeline predates the out-of-repo `preprocess_audio` baking,
so the int8 models run only via the streaming `feed()` path (`--streaming`).
