// VoixfulKit — the resident-model load entry point.
//
// This is the one place that turns a model URL (+ optional forced backend) into
// a warm, resident model, mirroring the CLI/README auto-detect flow. It lives in
// the umbrella (not the pure core) because it must reach every backend to
// dispatch through `BackendRegistry`.

import Foundation
// Re-export the resident-model surface so `import VoixfulKit` alone gives a host
// app everything it needs — `ResidentModel`, `OffloadPolicy`, `SpeechModelState`,
// and the returned `any Transcribing` — without a second `import VoixfulSpeech`.
@_exported import VoixfulSpeech
import VoixfulEngine

/// Load + graph-specialize a model once and keep it warm for reuse across many
/// dictation utterances (see `ResidentModel`). Warm-on-load by default: when this
/// returns the model is `.ready`. This free function is the single public entry
/// point for obtaining a resident model.
///
/// (A top-level function rather than `ResidentModel.load` because Swift can't
/// call a static member on the `ResidentModel` protocol metatype — the type
/// callers hold — so the loader lives beside it as a free function.)
///
/// - Parameters:
///   - model: the `.aimodel` directory.
///   - locale: BCP-47 locale (default en-US).
///   - backend: force the architecture; `nil` auto-detects from the asset
///     (matches the CLI/README).
///   - offload: idle + memory-pressure policy; `.pinned` disables both.
///   - piecesURL: SentencePiece vocab for parakeet/cohere/granite/canary;
///     defaults to the sibling `pieces.txt` bundled inside the asset.
///   - refreshStrideMS: live-partial refresh cadence for the vended
///     transcribers (default 700 ms). The app sets this per model to honor its
///     interval policy; ignored by the streaming (Nemotron) backend.
///   - commitThresholdSeconds: live-buffer size that triggers a head commit
///     (default 24 s); clamped down to the engine window at prepare.
public func loadResidentModel(
    model: URL,
    locale: Locale = Locale(identifier: "en-US"),
    backend: ModelBackend? = nil,
    offload: OffloadPolicy = .default,
    piecesURL: URL? = nil,
    refreshStrideMS: Int = 700,
    commitThresholdSeconds: Int = 24
) async throws -> any ResidentModel {
    let resolved = try backend ?? ModelBackend.detect(modelURL: model)
    let pieces = piecesURL ?? model.appendingPathComponent("pieces.txt")
    let ctx = ResidentContext(
        modelURL: model, piecesURL: pieces, locale: locale, offload: offload,
        refreshStrideMS: refreshStrideMS, commitThresholdSeconds: commitThresholdSeconds)
    return try await BackendRegistry.entry(for: resolved).loadResident(ctx)
}
