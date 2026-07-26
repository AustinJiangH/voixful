// VoixfulNemotronStreaming — residency for the Nemotron streaming backend.
//
// `ResidentNemotronModel` is the shared `ResidentModelActor` lifecycle wired to
// `NemotronStreamingEngine`: load = specialize the asset, warm = `engine.start()`
// (SOS reset), and each vended transcriber is a `NemotronStreamingTranscriber`
// driving the SHARED engine. Residency here targets live dictation, so it
// defaults to `.streaming` with volatile partials — mirroring `voixful mic`.

import Foundation
import VoixfulSpeech
import VoixfulEngine

/// Backend seam for the Nemotron streaming engine.
package struct NemotronResidentBackend: ResidentModelBackend {
    let modelURL: URL
    let language: Language
    let chunkSize: ChunkSize
    let chunkSizeMS: Int
    let mode: ProcessingMode
    let locale: Locale
    let reportingOptions: ReportingOption
    let debug: Bool

    package func load() async throws -> NemotronStreamingEngine {
        try await NemotronStreamingEngine(
            modelURL: modelURL, language: language, chunkSize: chunkSize)
    }

    package func warm(_ engine: NemotronStreamingEngine) async {
        // Nemotron warms via the streaming SOS reset rather than a silent buffer.
        await engine.start()
    }

    package func makeTranscriber(_ engine: NemotronStreamingEngine, locale: Locale?) -> NemotronStreamingTranscriber {
        // The engine is language-specific; a per-utterance locale override that
        // changed the language would be wrong on the shared engine, so we keep
        // the model's configured locale. Force-try is safe: language + chunk size
        // were validated when the resident model was built.
        try! NemotronStreamingTranscriber(
            engine: engine,
            locale: self.locale,
            chunkSizeMS: chunkSizeMS,
            mode: mode,
            reportingOptions: reportingOptions,
            attributeOptions: [],
            debug: debug)
    }
}

/// A resident, warm, auto-offloading Nemotron streaming model. Vends
/// `NemotronStreamingTranscriber`s over one shared engine.
package typealias ResidentNemotronModel = ResidentModelActor<NemotronResidentBackend>

extension ResidentModelActor where Backend == NemotronResidentBackend {
    /// Construct a Nemotron resident model from an asset on disk.
    package init(
        modelURL: URL,
        locale: Locale = Locale(identifier: "en-US"),
        chunkSizeMS: Int = 320,
        mode: ProcessingMode = .streaming,
        reportingOptions: ReportingOption = [.volatileResults],
        debug: Bool = false,
        offload: OffloadPolicy = .default,
        clock: any Clock<Duration> = ContinuousClock()
    ) throws {
        let tag = locale.identifier(.bcp47)
        guard let language = Language(rawValue: tag) else {
            throw VoixfulError.unsupportedLocale(tag)
        }
        guard let chunkSize = ChunkSize(rawValue: "\(chunkSizeMS)ms") else {
            throw VoixfulError.engine(
                "unsupported chunk size \(chunkSizeMS) ms (use 80, 160, 320, 560, 1120)")
        }
        self.init(
            backend: NemotronResidentBackend(
                modelURL: modelURL,
                language: language,
                chunkSize: chunkSize,
                chunkSizeMS: chunkSizeMS,
                mode: mode,
                locale: locale,
                reportingOptions: reportingOptions,
                debug: debug),
            offload: offload,
            clock: clock)
    }
}
