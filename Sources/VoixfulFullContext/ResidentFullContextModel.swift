// VoixfulFullContext — residency for the full-context (batch) backends.
//
// `ResidentFullContextModel<Engine>` is the shared `ResidentModelActor` lifecycle
// wired to a `FullContextEngine`: load = open asset + prewarm, warm = run a 0.5 s
// silent buffer to pay first-pass specialization, and each vended transcriber is
// a `LiveFullContextTranscriber` built around the SHARED engine.
//
// warmUp() window caveat: full-context engines are window-laddered
// (Parakeet/Cohere/Granite), so a 0.5 s silent buffer only specializes the
// SMALLEST window — a long first dictation may still specialize a larger window
// on its first pass. Whisper's fixed 30 s window is fully specialized by one
// silent pass.

import Foundation
import VoixfulSpeech

/// Backend seam for any `FullContextEngine` — the three operations the generic
/// resident lifecycle needs.
package struct FullContextResidentBackend<Engine: FullContextEngine>: ResidentModelBackend {
    let modelURL: URL
    let piecesURL: URL
    let locale: Locale
    let refreshStrideMS: Int
    let commitThresholdSeconds: Int

    package func load() async throws -> Engine {
        let engine = try await Engine(modelURL: modelURL, piecesURL: piecesURL)
        await engine.prewarm()   // background weight materialization
        return engine
    }

    package func warm(_ engine: Engine) async {
        // 0.5 s @ 16 kHz of silence → pay first-pass graph specialization off the
        // hot path. Best-effort: a warm failure just means the first real pass
        // eats the cost instead.
        let silent = [Float](repeating: 0, count: 8_000)
        _ = try? await engine.transcribe(audio: silent)
    }

    package func makeTranscriber(_ engine: Engine, locale: Locale?) -> LiveFullContextTranscriber<Engine> {
        LiveFullContextTranscriber(
            engine: engine,
            piecesURL: piecesURL,
            locale: locale ?? self.locale,
            refreshStrideMS: refreshStrideMS,
            commitThresholdSeconds: commitThresholdSeconds)
    }
}

/// A resident, warm, auto-offloading full-context model (Parakeet/Cohere/
/// Granite/Canary/Whisper). Vends `LiveFullContextTranscriber`s over one shared
/// engine.
package typealias ResidentFullContextModel<Engine: FullContextEngine> =
    ResidentModelActor<FullContextResidentBackend<Engine>>

extension ResidentModelActor {
    /// Construct a full-context resident model from an asset on disk.
    package init<Engine: FullContextEngine>(
        modelURL: URL,
        piecesURL: URL,
        locale: Locale = Locale(identifier: "en-US"),
        refreshStrideMS: Int = 700,
        commitThresholdSeconds: Int = 24,
        offload: OffloadPolicy = .default,
        clock: any Clock<Duration> = ContinuousClock()
    ) where Backend == FullContextResidentBackend<Engine> {
        self.init(
            backend: FullContextResidentBackend<Engine>(
                modelURL: modelURL,
                piecesURL: piecesURL,
                locale: locale,
                refreshStrideMS: refreshStrideMS,
                commitThresholdSeconds: commitThresholdSeconds),
            offload: offload,
            clock: clock)
    }
}
