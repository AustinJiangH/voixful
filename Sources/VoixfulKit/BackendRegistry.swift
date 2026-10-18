// VoixfulKit — the umbrella that imports every backend and holds the single
// data-driven registry mapping a `ModelBackend` to its constructors. This is the
// CLI's old position (it imported all backends and switched on the detected
// backend twice); consolidating those switches here means adding a backend is
// one registry row, not edits scattered across the CLI.
//
// `BackendRegistry.entry(for:)` returns three closures per backend:
//   • makeBatch    — a file/one-shot `Transcribing` (was the `Transcribe` switch)
//   • makeLive     — a live-mic `Transcribing`      (was the `Mic` switch)
//   • loadResident — a warm, reusable `ResidentModel` (new — residency)
//
// The `switch` is exhaustive over `ModelBackend`, so a new enum case fails to
// compile until its row is added — the property that keeps the registry honest.

import Foundation
import VoixfulSpeech
import VoixfulEngine
import VoixfulFullContext
import VoixfulNemotronStreaming
import VoixfulParakeetTDT
import VoixfulCohereTranscribe
import VoixfulGraniteNAR
import VoixfulGraniteTurboCTC
import VoixfulCanaryQwen
import VoixfulWhisper
import VoixfulQwen3ASR

/// Inputs a batch (file/one-shot) transcriber needs.
public struct BatchContext: Sendable {
    public let modelURL: URL
    public let piecesURL: URL
    public let locale: Locale
    /// Asset functions (read once by the caller) — drives Nemotron's
    /// batch-vs-streaming choice.
    public let fns: ModelBackend.AssetFunctions
    public let chunkSizeMS: Int
    public let bullets: Bool

    public init(modelURL: URL, piecesURL: URL, locale: Locale,
                fns: ModelBackend.AssetFunctions, chunkSizeMS: Int, bullets: Bool) {
        self.modelURL = modelURL
        self.piecesURL = piecesURL
        self.locale = locale
        self.fns = fns
        self.chunkSizeMS = chunkSizeMS
        self.bullets = bullets
    }
}

/// Inputs a live (microphone) transcriber needs.
public struct LiveContext: Sendable {
    public let modelURL: URL
    public let piecesURL: URL
    public let locale: Locale
    public let chunkSizeMS: Int
    public let debug: Bool

    public init(modelURL: URL, piecesURL: URL, locale: Locale, chunkSizeMS: Int, debug: Bool) {
        self.modelURL = modelURL
        self.piecesURL = piecesURL
        self.locale = locale
        self.chunkSizeMS = chunkSizeMS
        self.debug = debug
    }
}

/// Inputs a resident (warm, reusable) model needs.
public struct ResidentContext: Sendable {
    public let modelURL: URL
    public let piecesURL: URL
    public let locale: Locale
    public let offload: OffloadPolicy
    /// Live-partial refresh cadence and head-commit threshold for the vended
    /// full-context transcribers. Default to the transcriber defaults so the
    /// resident path preserves the app's per-model interval policy. Ignored by
    /// the streaming (Nemotron) row.
    public let refreshStrideMS: Int
    public let commitThresholdSeconds: Int

    public init(modelURL: URL, piecesURL: URL, locale: Locale, offload: OffloadPolicy,
                refreshStrideMS: Int = 700, commitThresholdSeconds: Int = 24) {
        self.modelURL = modelURL
        self.piecesURL = piecesURL
        self.locale = locale
        self.offload = offload
        self.refreshStrideMS = refreshStrideMS
        self.commitThresholdSeconds = commitThresholdSeconds
    }
}

/// The three constructors for one backend.
public struct BackendEntry {
    public let makeBatch: (BatchContext) throws -> any Transcribing
    public let makeLive: (LiveContext) throws -> any Transcribing
    public let loadResident: (ResidentContext) async throws -> any ResidentModel
}

public enum BackendRegistry {

    /// The registry row for `backend`. Exhaustive `switch` → adding a
    /// `ModelBackend` case is a compile error until its row lands here.
    public static func entry(for backend: ModelBackend) -> BackendEntry {
        switch backend {
        case .nemotron:
            return BackendEntry(
                makeBatch: { ctx in
                    // Batch needs the baked `preprocess_audio`; without it, stream.
                    try NemotronStreamingTranscriber(
                        modelURL: ctx.modelURL, locale: ctx.locale, chunkSizeMS: ctx.chunkSizeMS,
                        mode: ctx.fns.names.contains("preprocess_audio") ? .batch : .streaming,
                        reportingOptions: [], attributeOptions: [.audioTimeRange])
                },
                makeLive: { ctx in
                    try NemotronStreamingTranscriber(
                        modelURL: ctx.modelURL, locale: ctx.locale, chunkSizeMS: ctx.chunkSizeMS,
                        mode: .streaming, reportingOptions: [.volatileResults],
                        attributeOptions: [], debug: ctx.debug)
                },
                loadResident: { ctx in
                    let model = try ResidentNemotronModel(
                        modelURL: ctx.modelURL, locale: ctx.locale, offload: ctx.offload)
                    await model.warmUp()
                    return model
                })

        case .parakeet:
            return fullContext(
                ParakeetTDTEngine.self,
                batch: { ctx in ParakeetTDTTranscriber(
                    modelURL: ctx.modelURL, piecesURL: ctx.piecesURL, locale: ctx.locale) })

        case .cohere:
            return fullContext(
                CohereTranscribeEngine.self,
                batch: { ctx in CohereTranscriber(
                    modelURL: ctx.modelURL, piecesURL: ctx.piecesURL, locale: ctx.locale) })

        case .granite:
            return fullContext(
                GraniteNAREngine.self,
                batch: { ctx in GraniteTranscriber(
                    modelURL: ctx.modelURL, piecesURL: ctx.piecesURL, locale: ctx.locale) })

        case .turboctc:
            return fullContext(
                TurboCTCEngine.self,
                batch: { ctx in TurboCTCTranscriber(
                    modelURL: ctx.modelURL, piecesURL: ctx.piecesURL, locale: ctx.locale) })

        case .canary:
            return fullContext(
                CanaryQwenEngine.self,
                batch: { ctx in
                    if ctx.bullets { setenv("VOIXFUL_CANARY_FORMAT", "bullets", 1) }
                    return CanaryTranscriber(
                        modelURL: ctx.modelURL, piecesURL: ctx.piecesURL, locale: ctx.locale)
                })

        case .whisper:
            return fullContext(
                WhisperEngine.self,
                batch: { ctx in WhisperTranscriber(
                    modelURL: ctx.modelURL, piecesURL: ctx.piecesURL, locale: ctx.locale) })

        case .qwen3:
            return fullContext(
                Qwen3ASREngine.self,
                batch: { ctx in Qwen3ASRTranscriber(
                    modelURL: ctx.modelURL, piecesURL: ctx.piecesURL, locale: ctx.locale) })
        }
    }

    /// Shared row for a full-context backend: live + resident are identical
    /// across them (they differ only in the engine); only the batch transcriber
    /// varies, so it's supplied per-backend.
    private static func fullContext<Engine: FullContextEngine>(
        _ engine: Engine.Type,
        batch: @escaping (BatchContext) -> any Transcribing
    ) -> BackendEntry {
        BackendEntry(
            makeBatch: batch,
            makeLive: { ctx in
                LiveFullContextTranscriber<Engine>(
                    modelURL: ctx.modelURL, piecesURL: ctx.piecesURL, locale: ctx.locale)
            },
            loadResident: { ctx in
                let model = ResidentFullContextModel<Engine>(
                    modelURL: ctx.modelURL, piecesURL: ctx.piecesURL,
                    locale: ctx.locale,
                    refreshStrideMS: ctx.refreshStrideMS,
                    commitThresholdSeconds: ctx.commitThresholdSeconds,
                    offload: ctx.offload)
                await model.warmUp()
                return model
            })
    }
}
