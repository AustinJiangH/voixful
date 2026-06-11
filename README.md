# NemotronASR — Nemotron 3.5 ASR on Core AI

> **Goal** (from `PLAN.md`): End-to-end streaming transcription on Apple Silicon using **Apple Core AI** (the WWDC 2026 framework that supersedes Core ML for custom models) as the runtime for `nvidia/nemotron-3.5-asr-streaming-0.6b`. Swift host, PyTorch-based conversion, single `.aimodel` artifact, sub-realtime on M2.

**Status:** pre-spike scaffold; Phase 1+ pending.

## Repo layout

```
NemotronASR/
├── Package.swift
├── PLAN.md                     # source-of-truth design doc (mirror of canonical Obsidian copy)
├── HANDOFF.md                  # current state + what to do next
├── REVIEW_LOG.md               # per-phase review history (Phases A-H)
├── Sources/
│   ├── NemotronASR/            # library: mel, sessions, decoder, tokenizer, streaming
│   └── NemotronASRCLI/         # `nemotron-asr` executable (file + mic + languages subcommands)
├── Tests/
│   └── NemotronASRTests/       # unit + mel parity + cache round-trip + WER gate
├── convert/                    # Python conversion env (NeMo + coreai_torch)
│   ├── pyproject.toml
│   ├── README.md
│   └── nemotron_convert/       # extract / models / convert / export / verify_parity
├── Demos/
│   └── NemotronDemo/           # SwiftUI macOS demo app (local-path dep on this package)
└── fixtures/                   # sample.wav, mel_reference.bin, vocab.txt, aimodel/ (gitignored except sample.wav)
    └── README.md               # fixture generation recipes
```

## Where to look

- **Design** — read [`PLAN.md`](./PLAN.md). It is the source of truth for goals, package layout, API shape, anti-patterns, and the phased Phase 1+ task list (§7).
- **Current state and next steps** — read [`HANDOFF.md`](./HANDOFF.md). It enumerates every `// VERIFY` breadcrumb, the first three things to do, known gaps, and the commands to run once Phase 1 succeeds.
- **Per-phase review history** — read [`REVIEW_LOG.md`](./REVIEW_LOG.md). The `## Summary` block at the top consolidates blockers, fix-before-running, and fix-during-Phase-1 items across all phases.

## License

- **Model weights** (`nvidia/nemotron-3.5-asr-streaming-0.6b`): OpenMDW-1.1 — must be reviewed before redistribution (PLAN.md §10 open question 6).
- **Code in this repo:** TODO — set once project leaves spike phase. Suggest MIT for the Swift host and Python conversion utilities.
