# Contributing to Voixful

Thanks for your interest! Voixful is a SpeechAnalyzer-shaped, on-device ASR API
for Apple Silicon. Issues and pull requests are welcome.

## Ground rules

- Be respectful and constructive.
- Keep the **code / model-weights license split** intact: repository source is
  Apache-2.0; each `.aimodel` inherits its upstream checkpoint's license (see
  [`NOTICE`](NOTICE), [`MODEL_CARD.md`](MODEL_CARD.md), and the per-model
  `convert/*/README.md`). Never commit `.aimodel`, `.nemo`, `.pt`, or audio
  fixtures — they are gitignored on purpose.
- Don't add a dependency on Core ML / `coremltools`; Voixful targets Core AI.

## Requirements

- Apple Silicon Mac, macOS 27+, Swift 6 toolchain (Xcode 27).
- A converted `.aimodel` for the backend you're touching (download from Hugging
  Face or build with `convert/`).

## Build & test loop

```sh
swift build -c release       # build the package + `voixful` CLI
swift test                   # unit tests; fixture-gated tests XCTSkip on a
                             # clean checkout (see fixtures/README.md)
scripts/verify-backends.sh   # end-to-end smoke across installed backends
```

Please run `swift build` and `swift test` before opening a PR. If your change
affects a backend's runtime behavior, include a before/after transcript or timing
from `scripts/verify-backends.sh`.

## Adding a new backend

The analyzer only knows the `Transcribing` protocol, so a new model is
self-contained:

1. **Conversion** — add `convert/<model>/` with its own `pyproject.toml` and a
   `<model>_convert/` package exposing a `convert.py` that emits a `.aimodel`
   (support `--quantize {none,palette4,palette6}` where feasible). Keep deps
   isolated — do not share a venv across converters. Add a per-model `README.md`
   and note the upstream weights license.
2. **Swift backend** — add a `Voixful<Model>` target with a `<Model>Transcriber`
   conforming to `Transcribing`, plus its engine under
   `Sources/VoixfulEngine/<Model>/`. Register it in `Package.swift`.
3. **Detection** — extend `ModelBackend` (`Sources/VoixfulEngine/Shared/BackendDetection.swift`)
   so the CLI auto-detects the new asset from its function signatures.
4. **Docs** — add a row to the backend table in the top-level `README.md`.

## Commit & PR style

- Focused commits with clear messages; keep unrelated changes out of a PR.
- Describe what changed and how you verified it. Link any relevant issue.
