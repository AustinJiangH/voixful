// VoixfulSpeech — the orchestrator. Mirrors Apple's `SpeechAnalyzer`.
//
// Holds a set of plug-and-play modules, pumps an input audio sequence into all
// of them, and finalizes them together. Model-agnostic: it knows only the
// `VoixfulSpeechModule` protocol, so any backend's transcriber drops in.

import Foundation

public actor VoixfulAnalyzer {
    private let modules: [any VoixfulSpeechModule]
    private var pump: Task<Void, Error>?

    /// Construct with the modules to run (e.g. `[NemotronStreamingTranscriber(...)]`).
    /// Mirrors `SpeechAnalyzer(modules:)`.
    public init(modules: [any VoixfulSpeechModule]) {
        self.modules = modules
    }

    /// Begin analysis, pulling audio from `inputSequence` until it finishes.
    ///
    /// Prepares every module, then spawns a pump task that forwards each
    /// `AnalyzerInput` to all modules in order. Returns as soon as the pump is
    /// running; the audio is consumed asynchronously. Mirrors
    /// `SpeechAnalyzer.start(inputSequence:)`.
    public func start<S>(inputSequence: S) async throws
    where S: AsyncSequence, S: Sendable, S.Element == AnalyzerInput {
        for module in modules {
            try await module.prepare()
        }
        let modules = self.modules
        pump = Task {
            for try await input in inputSequence {
                for module in modules {
                    try await module.consume(input)
                }
            }
        }
    }

    /// Wait for the input sequence to drain, then finalize every module so they
    /// emit their trailing final results and close their streams. Mirrors
    /// `SpeechAnalyzer.finalizeAndFinishThroughEndOfInput()`.
    public func finalizeAndFinishThroughEndOfInput() async throws {
        try await pump?.value
        pump = nil
        for module in modules {
            try await module.finalize()
        }
    }

    /// Stop immediately without finalizing. Mirrors
    /// `SpeechAnalyzer.cancelAndFinishNow()`.
    public func cancelAndFinishNow() async {
        pump?.cancel()
        pump = nil
        for module in modules {
            await module.cancel()
        }
    }
}
