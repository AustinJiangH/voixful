# NemotronASR — Nemotron 3.5 ASR on Core AI

> **Goal** (from `PLAN.md`): End-to-end streaming transcription on Apple Silicon using **Apple Core AI** (the WWDC 2026 framework that supersedes Core ML for custom models) as the runtime for `nvidia/nemotron-3.5-asr-streaming-0.6b`. Swift host, PyTorch-based conversion, single `.aimodel` artifact, sub-realtime on M2.

## Try the demo

```sh
./scripts/demo.sh                       # live mic — speak; transcript updates in place
./scripts/demo.sh sample                # one-shot transcribe of a bundled "quick brown fox" sample
./scripts/demo.sh file path/to/audio.wav
```

First run downloads the model (~600 MB) and Python deps (~3 GB) and takes 5–10 minutes. Subsequent runs launch in seconds. On macOS the first mic launch triggers a permission prompt; grant it once and it stays granted.

The demo uses **NeMo's in-process inference** as the path-of-least-resistance live demo while the Swift + Core AI runtime catches up — see "Status" below. The Core AI conversion pipeline (`scripts/run-all.sh` → `python -m nemotron_convert.convert`) does work end-to-end and produces a 2.3 GB `.aimodel`; what's still pending is the Swift-side glue to drive it from a microphone.

## Status

| Track | State |
|---|---|
| Python conversion: NeMo → PyTorch → coreai_torch → `.aimodel` | ✅ Works on macOS 27 / Xcode 27 SDK / Python 3.12 / uv |
| `.aimodel` loads via `coreai.runtime.AIModel` with all three named functions | ✅ Verified |
| Live mic transcription via NeMo (the demo above) | ✅ Works |
| File transcription via NeMo (`demo.sh sample` / `demo.sh file`) | ✅ Works |
| Swift package builds + 30 tests pass + smoke driver green | ✅ |
| Swift + Core AI live mic (`nemotron-asr mic`, `Demos/NemotronDemo/`) end-to-end | 🚧 Phase 2 surgery in progress (see HANDOFF.md) |

## Repo layout

```
NemotronASR/
├── Package.swift
├── PLAN.md                     # source-of-truth design doc (mirror of canonical Obsidian copy)
├── HANDOFF.md                  # current state + what to do next
├── REVIEW_LOG.md               # per-phase review history
├── Sources/
│   ├── NemotronASR/            # library: mel, sessions, decoder, tokenizer, streaming
│   └── NemotronASRCLI/         # `nemotron-asr` executable (file + mic + smoke + languages)
├── Tests/
│   └── NemotronASRTests/       # unit + mel parity + cache round-trip + WER gate
├── convert/                    # Python conversion env (NeMo + coreai_torch + mic_demo + file_demo)
├── Demos/
│   └── NemotronDemo/           # SwiftUI macOS demo app (local-path dep on this package)
├── scripts/
│   ├── demo.sh                 # live mic / file / sample demos (this is the one to run)
│   ├── run-all.sh              # the full pipeline: env setup, download, convert, build, test
│   ├── smoke.sh                # wiring sanity check (no model required)
│   └── README.md
└── fixtures/                   # sample.wav, aimodel/ (gitignored except sample.wav)
    └── README.md
```

## Where to look

- **Live demo** — `./scripts/demo.sh`.
- **Design** — [`PLAN.md`](./PLAN.md).
- **Current state and what's next** — [`HANDOFF.md`](./HANDOFF.md).
- **Per-phase review history** — [`REVIEW_LOG.md`](./REVIEW_LOG.md).

## License

- **Model weights** (`nvidia/nemotron-3.5-asr-streaming-0.6b`): OpenMDW-1.1 — must be reviewed before redistribution (PLAN.md §10 open question 6).
- **Code in this repo:** TODO — suggest MIT for the Swift host and Python conversion utilities.
