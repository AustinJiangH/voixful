---
name: add-backend
description: >-
  Add a new ASR model/backend to the Voixful package end-to-end. Use when asked to
  "add a model", "add a backend", "wire up <model> (Parakeet/Cohere/Granite/Canary/
  Whisper/etc.)", "support a new .aimodel", or "register a new transcriber". Covers
  the three registries that must be updated in lockstep (the ModelBackend enum +
  detection heuristic, Package.swift products/targets, and the CLI switches), the
  FullContextEngine protocol contract, the convert/ pipeline, and the docs/scripts.
---

# Adding a new backend to Voixful

Voixful auto-detects the backend from an `.aimodel`'s function signatures and runs
every model through one `Transcribing` façade. Adding a model means creating a
conversion pipeline + an engine, then registering it in **three places** that must
stay in sync. Miss one and you get: builds-but-never-detected (registry 1),
doesn't-compile (registry 2), or detected-but-CLI-can't-instantiate (registry 3).

## Step 0 — Pick the backend shape

There are two lifecycles. Choose before writing code:

- **Full-context (batch)** — the model transcribes a whole clip in one shot
  (TDT, AED, NAR, SALM…). **This is the common, easy path.** Conform the engine to
  `FullContextEngine` and you get **both** file transcription (`FullContextTranscriber`)
  **and** live mic (`LiveFullContextTranscriber`) for free — no per-backend lifecycle
  code. Parakeet, Cohere, Granite, and Canary are all this shape.
- **Streaming (cache-aware)** — the encoder carries cache states and emits partials
  *while* audio arrives (only Nemotron today). This owns a bespoke transcriber and
  lifecycle; it does **not** use `FullContextEngine`. Only take this path if the
  model is genuinely cache-aware streaming.

The rest of this guide assumes the full-context path unless noted.

## The three registries (must be updated together)

| # | File | What to add |
|---|------|-------------|
| 1 | `Sources/VoixfulEngine/Shared/BackendDetection.swift` | a `ModelBackend` enum case **and** a `classify(_:)` rule keyed on a distinctive asset function name |
| 2 | `Package.swift` | a `.library` product, a `.target`, **and** the target in `VoixfulCLI`'s `dependencies` |
| 3 | `Sources/VoixfulCLI/main.swift` | `import Voixful<Model>` + a `case` in **both** the `Transcribe` and `Mic` switches |

## Step-by-step

### 1. Conversion pipeline — `convert/<model>/`
Mirror an existing dir (e.g. `convert/granite-speech/`). Self-contained: own
`pyproject.toml`, own `<model>_convert/` package with a `convert.py` that emits an
`.aimodel`, own `README.md`. Keep deps isolated (own venv) — never share.
- Support `--quantize {none,palette4,palette6}` where feasible (palettization is
  the RAM+latency win; see `convert/README.md`).
- **The asset's function names are the detection key** (registry 1 reads them). Give
  the new model a *distinctive* entrypoint name so `classify` can tell it apart —
  e.g. Canary ships `step_asr`, Granite ships `edit_<A>_<U>`, Cohere ships
  `decode_step*`. Bundle `pieces.txt` (and any `<model>.json`/`tokens.json`) *inside*
  the `.aimodel` dir so the asset is self-contained.

### 2. Engine — `Sources/VoixfulEngine/<Model>/`
The Core AI runtime. For a full-context backend, make it an `actor` conforming to
`FullContextEngine` (`Sources/VoixfulFullContext/FullContextTranscriber.swift`):

```swift
public protocol FullContextEngine: Actor {
    init(modelURL: URL, piecesURL: URL) async throws
    func transcribe(audio: [Float]) async throws -> Transcript   // Transcript = VoixfulEngine/Shared/Types.swift
    func prewarm()                                                // background-load likely functions; return immediately
    static var displayName: String { get }
    static var identifierSuffix: String { get }                  // → com.voixful.transcriber.<suffix>
    static var supportedLocales: [Locale] { get }
}
```

Reuse `Shared/` infra (MelExtractor, SentencePieceTokenizer, NDArrayHelpers,
CPUTransducerDecoder, EncodeWindows). Nothing is shared *across* models except
`Shared/`, so odd deps don't leak.

### 3. Registry 1 — detection (`Sources/VoixfulEngine/Shared/BackendDetection.swift`)
Add the enum case and a `classify` rule. **Order matters** — most specific first,
because several backends share stateless `encode_<T>` functions:

```swift
public enum ModelBackend: String, Sendable, CaseIterable {
    case nemotron, parakeet, cohere, granite, canary, <model>
}
// in classify(_:), before the generic fallbacks:
if fns.names.contains("<distinctive_fn_name>") { return .<model> }
```
The fallback order today: `step_asr`→canary, `edit_`→granite, `decode_*`→cohere,
`encode` has states→nemotron, else→parakeet. Insert your rule where it won't be
shadowed.

### 4. Façade target — `Sources/Voixful<Model>/<Model>Transcriber.swift`
Tiny. Conform the engine and alias the shared generic (copy
`Sources/VoixfulParakeetTDT/ParakeetTDTTranscriber.swift`):

```swift
import VoixfulEngine
import VoixfulFullContext

extension <Model>Engine: FullContextEngine {
    public static let displayName = "<Human Name>"
    public static let identifierSuffix = "<model>"
    public static let supportedLocales = [Locale(identifier: "en-US")]
}
public typealias <Model>Transcriber = FullContextTranscriber<<Model>Engine>
```

### 5. Registry 2 — `Package.swift`
Three edits:
```swift
.library(name: "Voixful<Model>", targets: ["Voixful<Model>"]),          // products
.target(name: "Voixful<Model>",
        dependencies: ["VoixfulSpeech", "VoixfulEngine", "VoixfulFullContext"]),  // targets
// add "Voixful<Model>" to the VoixfulCLI executableTarget dependencies list
```

### 6. Registry 3 — CLI (`Sources/VoixfulCLI/main.swift`)
- `import Voixful<Model>` at the top.
- `Transcribe.run()` switch (batch): `case .<model>: transcriber = <Model>Transcriber(modelURL: modelURL, piecesURL: piecesURL(modelURL), locale: locale)`
- `Mic.run()` switch (live): `case .<model>: transcriber = LiveFullContextTranscriber<<Model>Engine>(modelURL: modelURL, piecesURL: piecesURL, locale: locale)`

### 7. Docs & scripts
- README.md: add a row to the backend table.
- `scripts/verify-backends.sh`: add the palette4 asset name + a `run "<model>" "$ASSET" "expected words"` line (it skips gracefully if the asset is absent).
- `convert/README.md`: add the dir to the tree + the upstream-weights license to the license table.
- `MODEL_CARD.md` if you publish weights.

## Verify

```sh
swift build && swift build --build-tests   # all three registries agree
swift test                                 # 0 failures (fixture-gated tests skip)
.build/debug/voixful transcribe --help     # --backend help lists the new case
scripts/verify-backends.sh ~/path/to/models # end-to-end if you have the .aimodel
```

A green `swift build` proves registries 2+3 are consistent; a passing
`ModelBackend.detect` on a real asset proves registry 1. If the CLI reports the
wrong backend, your `classify` rule is shadowed — reorder it.
