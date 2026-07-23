// Detect which decode runtime an `.aimodel` needs, from its function signatures.
//
// The right backend is a property of the asset, so the CLI shouldn't have to be
// told. We read it straight from the on-disk asset summary — NOT by specializing
// a runnable model (that would share a cached instance with the engine and
// corrupt the streaming encoder's state setup):
//
//   - Canary-Qwen (SALM): has a `step_asr` function (autoregressive Qwen3 decode).
//   - Granite (NAR): has `edit_<A>_<U>` single-pass LLM edit functions.
//   - Cohere (attention encoder-decoder): has a `decode_step*` function.
//   - Nemotron (streaming, cache-aware RNN-T): `encode` carries streaming cache
//     STATES (cache_last_channel/…), no `decode_step`.
//   - Parakeet (full-context TDT): `encode` has no states.

import CoreAI
import Foundation

public enum ModelBackend: String, Sendable, CaseIterable {
    case nemotron, parakeet, cohere, granite, canary
}

extension ModelBackend {
    /// The function names + their state names, read from the on-disk asset
    /// summary (metadata only — no specialization).
    public struct AssetFunctions: Sendable {
        public let names: Set<String>
        /// Whether `encode` carries cache states (streaming, cache-aware).
        public let encodeHasStates: Bool
    }

    public static func assetFunctions(modelURL: URL) throws -> AssetFunctions {
        let asset = try AIModelAsset(contentsOf: modelURL)
        guard let summary = try asset.summary(includingStatistics: false) else {
            throw ASRError.modelLoadFailed("no asset summary for \(modelURL.path)")
        }
        let names = Set(summary.functions.map { $0.name })
        let encodeHasStates = summary.functions
            .first(where: { $0.name == "encode" })
            .map { !$0.states.isEmpty } ?? false
        return AssetFunctions(names: names, encodeHasStates: encodeHasStates)
    }

    /// The single backend-classification rule, over already-read functions. Both
    /// `detect` and the CLI go through this so the heuristic lives in one place.
    /// Cohere is any AED decode entrypoint: lean assets ship only the cached
    /// decoders (decode_x4/decode_step_fused); `decode_step` is the legacy
    /// stateless program, exported only under `--legacy-decoders`.
    public static func classify(_ fns: AssetFunctions) -> ModelBackend {
        // Canary-Qwen SALM: AR Qwen3 step functions. Checked first — its
        // encode_<T> functions are stateless like Parakeet's.
        if fns.names.contains("step_asr") {
            return .canary
        }
        // Granite NAR: single-pass LLM edit entrypoints (edit_<A>_<U>).
        if fns.names.contains(where: { $0.hasPrefix("edit_") }) {
            return .granite
        }
        if !fns.names.isDisjoint(with: ["decode_x4", "decode_step_fused", "decode_step_cached", "decode_step"]) {
            return .cohere
        }
        if fns.encodeHasStates { return .nemotron }   // cache-aware streaming encoder
        return .parakeet
    }

    /// Classify the asset by its function signatures (reads metadata only).
    public static func detect(modelURL: URL) throws -> ModelBackend {
        classify(try assetFunctions(modelURL: modelURL))
    }
}
