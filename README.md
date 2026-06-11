# NemotronASR — Nemotron 3.5 ASR on Apple Core AI

End-to-end streaming transcription on Apple Silicon, running `nvidia/nemotron-3.5-asr-streaming-0.6b` through **Core AI** (the WWDC 2026 framework that succeeds Core ML for custom models). Pure Swift at runtime — the `.aimodel` artifact does all the work on the ANE/GPU.

## Try the demo

```sh
./scripts/demo.sh                       # live mic: speak; transcript appears per utterance
./scripts/demo.sh sample                # one-shot transcribe of the bundled sample.wav
./scripts/demo.sh file path/to/foo.wav  # one-shot transcribe of any WAV
```

First run builds the Swift CLI in release mode (~15 s). The first mic launch triggers a macOS permission prompt for `nemotron-asr`; grant it once.

## Performance

| Path | Runtime | 5.5 s sample.wav |
|---|---|---|
| Swift + Core AI (ANE/GPU) | ~1.4 s | **4.5× faster** |
| Python + NeMo on CPU (former demo path) | ~6.4 s | baseline |

## Status

| Capability | State |
|---|---|
| `swift run nemotron-asr file <wav>` via the `.aimodel` | ✅ produces a transcript |
| `swift run nemotron-asr mic` via the `.aimodel` | ✅ utterance-by-utterance live captions |
| Build cleanly on macOS 27 / Xcode 27 SDK / Swift 6.4 | ✅ |
| Swift tests | ✅ 30 pass, 5 fixture-skip |
| Mel preprocessor parity with NeMo (preemph + reflect-pad applied) | ✅ within ~0.5 nats |

## Repo layout

```
NemotronASR/
├── Package.swift
├── PLAN.md                     # source-of-truth design doc
├── HANDOFF.md                  # current state + open follow-ups
├── REVIEW_LOG.md               # per-phase review history (scaffolding workflows)
├── Sources/
│   ├── NemotronASR/            # library: mel, sessions, decoder, tokenizer, streaming
│   └── NemotronASRCLI/         # `nemotron-asr` executable
├── Tests/NemotronASRTests/
├── Demos/NemotronDemo/         # SwiftUI macOS demo app
├── scripts/
│   ├── demo.sh                 # the one to run
│   ├── run-all.sh              # build + test + sample transcription
│   └── smoke.sh                # fast wiring check
└── fixtures/
    ├── sample.wav              # bundled audio
    └── aimodel/
        ├── nemotron-streaming.aimodel  # the converted model (~2.3 GB)
        ├── pieces.txt          # SentencePiece vocab (1024 tokens + blank)
        ├── tokenizer.model     # raw SP binary
        └── vocab.txt           # WordPiece-style copy (unused by Swift)
```

## License

- **Model weights** (`nvidia/nemotron-speech-streaming-en-0.6b`): OpenMDW-1.1 — review before redistribution.
- **Code in this repo:** TODO — suggest MIT.
