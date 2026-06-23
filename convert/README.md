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

The default artifacts are **FP16, no weight quantization** (see `PLAN.md` §3
Step 5). `coreai_opt` INT4/INT8 weight-only quantization is the lever for cutting
encoder size/RAM (the encoder is ~85% of compute and the bulk of the on-disk
size) — see each model's `convert.py` step (e) for where the `coreai_opt` pass
slots in, and `PLAN.md` §3 Step 5b for the recipe.
