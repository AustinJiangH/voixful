// VoixfulKit — the resident-model load entry point.
//
// This is the one place that turns a model URL (+ optional forced backend) into
// a warm, resident model, mirroring the CLI/README auto-detect flow. It lives in
// the umbrella (not the pure core) because it must reach every backend to
// dispatch through `BackendRegistry`.

import Foundation
import VoixfulSpeech
import VoixfulEngine

/// Load + graph-specialize a model once and keep it warm for reuse across many
/// dictation utterances (see `ResidentModel`). Warm-on-load by default: when this
/// returns the model is `.ready`.
///
/// (A top-level function rather than `ResidentModel.load` because Swift can't
/// call a static member on a protocol metatype; `ResidentModel` is the vended
/// protocol, so the loader lives beside it as a free function. `ResidentModel`
/// also carries a `static load` usable on any concrete conformer.)
///
/// - Parameters:
///   - model: the `.aimodel` directory.
///   - locale: BCP-47 locale (default en-US).
///   - backend: force the architecture; `nil` auto-detects from the asset
///     (matches the CLI/README).
///   - offload: idle + memory-pressure policy; `.pinned` disables both.
///   - piecesURL: SentencePiece vocab for parakeet/cohere/granite/canary;
///     defaults to the sibling `pieces.txt` bundled inside the asset.
public func loadResidentModel(
    model: URL,
    locale: Locale = Locale(identifier: "en-US"),
    backend: ModelBackend? = nil,
    offload: OffloadPolicy = .default,
    piecesURL: URL? = nil
) async throws -> any ResidentModel {
    let resolved = try backend ?? ModelBackend.detect(modelURL: model)
    let pieces = piecesURL ?? model.appendingPathComponent("pieces.txt")
    let ctx = ResidentContext(
        modelURL: model, piecesURL: pieces, locale: locale, offload: offload)
    return try await BackendRegistry.entry(for: resolved).loadResident(ctx)
}

extension ResidentModel {
    /// Same as `loadResidentModel(...)`, callable on a concrete resident type
    /// (e.g. `ResidentFullContextModel<E>.load(...)`). The bare protocol metatype
    /// can't carry a static factory, so prefer the free `loadResidentModel(...)`.
    public static func load(
        model: URL,
        locale: Locale = Locale(identifier: "en-US"),
        backend: ModelBackend? = nil,
        offload: OffloadPolicy = .default,
        piecesURL: URL? = nil
    ) async throws -> any ResidentModel {
        try await loadResidentModel(
            model: model, locale: locale, backend: backend,
            offload: offload, piecesURL: piecesURL)
    }
}
