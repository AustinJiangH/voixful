# scripts/

> Code in this repo is Apache-2.0 (root `LICENSE`/`NOTICE`); model weights inherit
> their upstream model licenses (see `../MODEL_CARD.md` and the per-model
> `convert/*/README.md`).

Top-level driver scripts for the Voixful project. Each script is
self-contained, prints greppable status lines, and exits non-zero on hard failure.

## `smoke.sh`

Fast, **no-model** wiring sanity check.

**What it does**

The smallest end-to-end exercise that confirms "the wiring is sane" without any
`.aimodel` asset: it `swift build`s the package and runs `swift test` (model-free
tests run; fixture-gated tests `XCTSkip`), then points you at `verify-backends.sh`
for the real model-driven runs. Exits non-zero on build/test failure.

**Usage**

```bash
./scripts/smoke.sh
```

No arguments. Run from any directory; the script resolves `ROOT` from its own
location. Under ~10 s wall-clock once the build is warm.

## `verify-backends.sh`

End-to-end verification of the `voixful` CLI across **all three backends**
(Nemotron streaming RNN-T, Parakeet full-context TDT, Cohere AED). Builds release
if needed, runs each backend on `fixtures/sample.wav`, and checks the transcript.

**Usage**

```bash
./scripts/verify-backends.sh [MODELS_DIR]
```

`MODELS_DIR` defaults to `~/repos/voixful-models` and must contain the palette4
assets, each self-contained (`pieces.txt` bundled inside the `.aimodel` dir):
`nemotron-speech-streaming-en-0.6b-palette4.aimodel`,
`parakeet-tdt-0.6b-v3-palette4.aimodel`,
`cohere-transcribe-03-2026-palette4.aimodel`. Exits non-zero on any failure.

## `run-all.sh`

Builds the Swift package, runs tests, and transcribes the bundled
`fixtures/sample.wav` as a smoke check (no Python required). Use
`verify-backends.sh` for multi-backend coverage.

## Adding scripts

Drop additional drivers in this directory. Convention:
1. `#!/usr/bin/env bash` + `set -euo pipefail` header.
2. Resolve the repo root from `BASH_SOURCE[0]` so the script works from any cwd.
3. Print greppable status lines (a tool prefix or step marker per line).
4. Exit non-zero only on hard failure.
5. Document the script in this README.
