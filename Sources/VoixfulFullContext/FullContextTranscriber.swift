// VoixfulFullContext — the shared `Transcribing` implementation for full-context
// (batch) backends. Parakeet TDT and Cohere Transcribe differ only in their
// engine and a few metadata strings; everything else (buffer-all-via-consume,
// transcribe-once-at-finalize, the result/range plumbing) is identical. This one
// generic actor captures it so the two backends don't carry duplicate copies.
//
// A streaming backend (Nemotron) does NOT use this — it emits volatile partials
// per chunk and owns a different lifecycle.
//
// Option contract: batch backends emit exactly one final result at `finalize`, so
// `reportingOptions.volatileResults` is a no-op here (no partials). The utterance
// time span is always reported via `TranscriptionResult.range`; `attributeOptions`
// inline attributes are not attached.

import AVFoundation
import Foundation
import VoixfulEngine
import VoixfulSpeech

/// A batch ASR engine that loads from an `.aimodel` + pieces file and transcribes
/// a complete clip in one shot. `FullContextTranscriber` adapts any conformer to
/// the `Transcribing` façade. Engines return the shared `Transcript` (same final
/// result type as the streaming `NemotronStreamingEngine`).
public protocol FullContextEngine: Actor {
    init(modelURL: URL, piecesURL: URL) async throws
    func transcribe(audio: [Float]) async throws -> Transcript

    /// Kick off background loads of the inference functions a session will
    /// most likely need (returns immediately). Called by `prepare()` so weight
    /// materialization overlaps audio capture instead of the first transcribe.
    func prewarm()

    /// Human-readable model name (e.g. "Parakeet TDT 0.6B").
    static var displayName: String { get }
    /// Suffix for the reverse-DNS module id: `com.voixful.transcriber.<suffix>`.
    static var identifierSuffix: String { get }
    /// Locales this backend advertises.
    static var supportedLocales: [Locale] { get }

    /// The largest audio (16 kHz samples) the engine transcribes in ONE pass —
    /// the size of its FIXED, exported encoder window (Parakeet: 1000 mel
    /// frames = 10 s). This is NOT a length limit on what a user may dictate:
    /// callers transcribe longer audio by segmenting it into chunks of this
    /// size (see `LongForm`). It only bounds a single `transcribe` call.
    nonisolated var maxAudioSamples: Int { get }
}

public actor FullContextTranscriber<Engine: FullContextEngine>: Transcribing {

    // MARK: Transcribing / VoixfulSpeechModule
    public nonisolated let moduleIdentifier: String
    public nonisolated let modelName: String
    public nonisolated let locale: Locale
    public nonisolated let reportingOptions: ReportingOption
    public nonisolated let attributeOptions: ResultAttributeOption
    public nonisolated let results: AsyncThrowingStream<TranscriptionResult, Error>

    public static var supportedLocales: [Locale] { Engine.supportedLocales }

    private nonisolated let continuation: AsyncThrowingStream<TranscriptionResult, Error>.Continuation
    private let modelURL: URL
    private let piecesURL: URL
    private let converter = AudioInputConverter()
    private var engine: Engine?
    private var utterance: [Float] = []
    private var samplesSeen = 0
    private var utteranceStart: Double?

    public init(
        modelURL: URL,
        piecesURL: URL,
        locale: Locale = Locale(identifier: "en-US"),
        reportingOptions: ReportingOption = [],
        attributeOptions: ResultAttributeOption = []
    ) {
        self.modelURL = modelURL
        self.piecesURL = piecesURL
        self.moduleIdentifier = "com.voixful.transcriber.\(Engine.identifierSuffix)"
        self.modelName = Engine.displayName
        self.locale = locale
        self.reportingOptions = reportingOptions
        self.attributeOptions = attributeOptions
        let (stream, continuation) = AsyncThrowingStream<TranscriptionResult, Error>.makeStream()
        self.results = stream
        self.continuation = continuation
    }

    public func prepare() async throws {
        guard engine == nil else { return }
        do {
            engine = try await Engine(modelURL: modelURL, piecesURL: piecesURL)
        } catch {
            throw VoixfulError.engine("could not load \(Engine.displayName) engine: \(error)")
        }
        // Non-blocking: spawns background function loads and returns.
        await engine?.prewarm()
    }

    public func consume(_ input: AnalyzerInput) async throws {
        let samples = try converter.convert(input.buffer)
        if utteranceStart == nil { utteranceStart = input.bufferStartTime?.seconds ?? 0 }
        samplesSeen += samples.count
        utterance.append(contentsOf: samples)
    }

    public func finalize() async throws {
        guard let engine else { continuation.finish(); return }
        do {
            // Segment on silence into window-sized chunks so a clip longer than
            // the engine's fixed encoder window (Parakeet: 10 s) transcribes in
            // full instead of being truncated at the window. One pass when it
            // already fits.
            let text = try await LongForm.transcribe(audio: utterance, with: engine)
            let start = utteranceStart ?? 0
            let duration = Double(samplesSeen) / 16_000.0
            let range = CMTimeRange(
                start: CMTime(seconds: start, preferredTimescale: 1_000),
                duration: CMTime(seconds: duration, preferredTimescale: 1_000))
            continuation.yield(TranscriptionResult(
                text: AttributedString(text), range: range, isFinal: true))
            continuation.finish()
        } catch {
            let wrapped = VoixfulError.engine("finalize failed: \(error)")
            continuation.finish(throwing: wrapped)
            throw wrapped
        }
    }

    /// Tear down without finalizing (cancelAndFinishNow).
    public func cancel() async {
        continuation.finish()
    }
}
