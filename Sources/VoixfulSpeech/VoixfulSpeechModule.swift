// VoixfulSpeech — module protocols.
//
// `VoixfulSpeechModule` is the analyzer↔module contract (audio in). `Transcribing`
// refines it with the transcription result stream (results out). Concrete
// modules — `NemotronStreamingTranscriber` and future backends — live in their own
// targets and conform to `Transcribing`. This is what makes models swappable
// under one brand: the analyzer only knows the protocol.

import Foundation

/// Base capability every Voixful module exposes to the analyzer.
///
/// The `consume`/`finalize`/`cancel` methods are driven by `VoixfulAnalyzer`,
/// not by application code — callers interact with a module through its
/// result stream (e.g. `Transcribing.results`), never by feeding it audio
/// directly.
public protocol VoixfulSpeechModule: Sendable {
    /// Stable reverse-DNS identifier, e.g. `"com.voixful.transcriber.nemotron"`.
    var moduleIdentifier: String { get }

    /// Load/warm any resources. Called once by the analyzer on `start`.
    func prepare() async throws

    /// Consume one input buffer. Called by the analyzer for each `AnalyzerInput`.
    func consume(_ input: AnalyzerInput) async throws

    /// Flush remaining audio and emit final results. Called by the analyzer on
    /// `finalizeAndFinishThroughEndOfInput()`.
    func finalize() async throws

    /// Tear down without finalizing. Called by `cancelAndFinishNow()`.
    func cancel() async
}

/// A module that turns audio into text. Mirrors `SpeechTranscriber`.
public protocol Transcribing: VoixfulSpeechModule {
    /// Human-readable model name for display, e.g. "Nemotron". Surfaced by UIs
    /// (the live mic TUI shows it next to the Voixful brand) so each backend
    /// self-identifies.
    var modelName: String { get }

    /// Locales this backend can transcribe.
    static var supportedLocales: [Locale] { get }

    /// The locale this instance was configured with.
    var locale: Locale { get }

    /// Reporting options (volatile results, alternatives, …).
    var reportingOptions: ReportingOption { get }

    /// Per-run attribute options (audio time range, confidence, …).
    var attributeOptions: ResultAttributeOption { get }

    /// The live result stream. Read this to receive volatile + final results.
    var results: AsyncThrowingStream<TranscriptionResult, Error> { get }
}
