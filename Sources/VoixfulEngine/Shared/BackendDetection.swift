// Detect which decode runtime an `.aimodel` needs, from its function signatures.
//
// The right backend is a property of the asset, so the CLI shouldn't have to be
// told. We read it straight from the on-disk asset summary — NOT by specializing
// a runnable model (that would share a cached instance with the engine and
// corrupt the streaming encoder's state setup):
//
//   - Qwen3-ASR (SALM): has a `step_qwen3` function (autoregressive Qwen3 decode,
//     Alibaba/Qwen). Distinct entrypoint name keeps it apart from Canary's `step_asr`.
//   - Canary-Qwen (SALM): has a `step_asr` function (autoregressive Qwen3 decode).
//   - Granite (NAR): has `edit_<A>_<U>` single-pass LLM edit functions.
//   - Whisper (attention encoder-decoder): has a `whisper_decode` function (the
//     distinctive name keeps it apart from Cohere's `decode_step*`).
//   - Cohere (attention encoder-decoder): has a `decode_step*` function.
//   - Nemotron (streaming, cache-aware RNN-T): `encode` carries streaming cache
//     STATES (cache_last_channel/…), no `decode_step`.
//   - Parakeet (full-context TDT): `encode` has no states.

import CoreAI
import Foundation

public enum ModelBackend: String, Sendable, CaseIterable {
    case nemotron, parakeet, cohere, granite, canary, whisper, qwen3

    /// Longest audio the backend's standard converted asset transcribes in ONE
    /// pass — the SMALLER of its fixed encoder window and any downstream decoder
    /// token cap (the same quantity each engine exposes at runtime as
    /// `maxAudioSamples`, here as static registry metadata / 16 kHz seconds):
    ///   • parakeet — 30 s (3000-frame mel encoder window; asset carries the exact
    ///                value in parakeet.json — older 1000-frame assets are 10 s)
    ///   • cohere   — 15 s (108-token AED decoder cap binds before the 35 s window)
    ///   • granite  — 32 s (largest 1600-frame encoder window; token budget scales)
    ///   • canary   — 36 s (256-token decode cap binds before the 40 s window)
    ///   • qwen3    — 30 s (audio tower's fixed 30 s Whisper window / 3000 mel
    ///                frames; converted asset ships a single encode_3000)
    ///   • whisper  — 30 s (Whisper's fixed 30 s encoder window; audio is always
    ///                padded/truncated to it)
    ///   • nemotron — nil  (cache-aware streaming; no window, unbounded)
    ///
    /// This is NOT a limit on what a user may dictate: the full-context
    /// transcribers segment longer audio automatically (see VoixfulFullContext
    /// `LongForm`). It's an informational hint for consumers — e.g. a live HUD
    /// showing where a single re-transcribe pass tops out.
    public var singlePassWindowSeconds: Double? {
        switch self {
        case .nemotron: return nil
        case .parakeet: return 30
        case .cohere:   return 15
        case .granite:  return 32
        case .canary:   return 36
        case .qwen3:    return 30
        case .whisper:  return 30
        }
    }
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
        // Qwen3-ASR SALM (Alibaba/Qwen): AR Qwen3 step function. Distinct
        // entrypoint (`step_qwen3`) so it isn't confused with Canary's `step_asr`;
        // like Canary its encode_<T> functions are stateless (would else fall
        // through to Parakeet), so it's checked up here before the fallbacks.
        if fns.names.contains("step_qwen3") {
            return .qwen3
        }
        // Canary-Qwen SALM: AR Qwen3 step functions. Checked first — its
        // encode_<T> functions are stateless like Parakeet's.
        if fns.names.contains("step_asr") {
            return .canary
        }
        // Granite NAR: single-pass LLM edit entrypoints (edit_<A>_<U>).
        if fns.names.contains(where: { $0.hasPrefix("edit_") }) {
            return .granite
        }
        // Whisper AED: distinctively-named decode entrypoint. Checked BEFORE
        // Cohere so the generic `decode_*` rule below can't shadow it.
        if fns.names.contains("whisper_decode") {
            return .whisper
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
