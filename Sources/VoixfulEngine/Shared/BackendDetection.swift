// Detect which decode runtime an `.aimodel` needs, from its function signatures.
//
// The right backend is a property of the asset, so the CLI shouldn't have to be
// told. We read it straight from the on-disk asset summary — NOT by specializing
// a runnable model (that would share a cached instance with the engine and
// corrupt the streaming encoder's state setup):
//
//   - Cohere (attention encoder-decoder): has a `decode_step` function.
//   - Nemotron (streaming, cache-aware RNN-T): `encode` carries streaming cache
//     STATES (cache_last_channel/…), no `decode_step`.
//   - Parakeet (full-context TDT): `encode` has no states.

import CoreAI
import Foundation

public enum ModelBackend: String, Sendable, CaseIterable {
    case nemotron, parakeet, cohere
}

extension ModelBackend {
    /// Classify the asset by its function signatures (reads metadata only).
    public static func detect(modelURL: URL) throws -> ModelBackend {
        let asset = try AIModelAsset(contentsOf: modelURL)
        guard let summary = try asset.summary(includingStatistics: false) else {
            throw ASRError.modelLoadFailed("no asset summary for \(modelURL.path)")
        }
        let functions = summary.functions
        if functions.contains(where: { $0.name == "decode_step" }) { return .cohere }
        if let encode = functions.first(where: { $0.name == "encode" }), !encode.states.isEmpty {
            return .nemotron   // cache-aware streaming encoder carries states
        }
        return .parakeet
    }
}
