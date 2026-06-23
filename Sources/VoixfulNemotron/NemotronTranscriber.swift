// VoixfulNemotron — the Nemotron 3.5 backend behind the Voixful brand surface.
//
// `NemotronTranscriber` conforms to `VoixfulSpeech.Transcribing`, wrapping the
// existing `StreamingASR` engine. Its public surface uses only Foundation +
// VoixfulSpeech types (Locale, ReportingOption, …) — the Nemotron-specific
// `Language`/`ChunkSize`/`StreamingASR` types stay internal, so the brand API
// is identical across backends.
//
// Implemented as an `actor` for its mutable per-utterance state, but the
// caller-facing surface (`results`, `locale`, `moduleIdentifier`, options) is
// exposed via `nonisolated let` so it reads synchronously, exactly like Apple's
// `SpeechTranscriber`.
//
// Streaming strategy (A) per DESIGN_ANALYZER_API.md §5: accumulate audio,
// emit throttled volatile results by re-transcribing the growing utterance,
// and one authoritative final result on `finalize()`. The true per-buffer
// streaming path is a fast-follow gated on mel parity.

import Foundation
import CoreMedia
import VoixfulSpeech
import NemotronASR

public actor NemotronTranscriber: Transcribing {

    // MARK: - Caller-facing surface (synchronous)

    public nonisolated let moduleIdentifier = "com.voixful.transcriber.nemotron"
    public nonisolated let locale: Locale
    public nonisolated let reportingOptions: ReportingOption
    public nonisolated let attributeOptions: ResultAttributeOption
    public nonisolated let results: AsyncThrowingStream<TranscriptionResult, Error>

    public nonisolated static var supportedLocales: [Locale] {
        Language.allCases.map { Locale(identifier: $0.rawValue) }
    }

    // MARK: - Internal state

    private nonisolated let continuation: AsyncThrowingStream<TranscriptionResult, Error>.Continuation
    private let modelURL: URL
    private let language: Language
    private let chunkSize: ChunkSize
    private let converter = AudioInputConverter()

    private var engine: StreamingASR?
    private var utterance: [Float] = []
    private var utteranceStart: Double?
    private var lastVolatileCount = 0

    /// Minimum new audio (samples @ 16 kHz) between volatile re-transcriptions.
    /// 8000 = 0.5 s; keeps the re-transcribe cost off the per-buffer hot path.
    private static let volatileStrideSamples = 8_000

    // MARK: - Init

    /// Create a transcriber. Model loading is deferred to `prepare()` (driven by
    /// the analyzer) so this initializer stays synchronous like Apple's.
    ///
    /// - Throws: `VoixfulError.unsupportedLocale` if `locale` is outside the
    ///   backend's `supportedLocales`, or `.engine` for an unsupported chunk size.
    public init(
        modelURL: URL,
        locale: Locale = Locale(identifier: "en-US"),
        chunkSizeMS: Int = 320,
        reportingOptions: ReportingOption = [],
        attributeOptions: ResultAttributeOption = []
    ) throws {
        let tag = locale.identifier(.bcp47)
        guard let language = Language(rawValue: tag) else {
            throw VoixfulError.unsupportedLocale(tag)
        }
        guard let chunkSize = ChunkSize(rawValue: "\(chunkSizeMS)ms") else {
            throw VoixfulError.engine(
                "unsupported chunk size \(chunkSizeMS) ms (use 80, 160, 320, 560, 1120)"
            )
        }

        self.modelURL = modelURL
        self.language = language
        self.locale = locale
        self.chunkSize = chunkSize
        self.reportingOptions = reportingOptions
        self.attributeOptions = attributeOptions

        let (stream, continuation) = AsyncThrowingStream<TranscriptionResult, Error>.makeStream()
        self.results = stream
        self.continuation = continuation
    }

    // MARK: - VoixfulSpeechModule

    public func prepare() async throws {
        guard engine == nil else { return }
        do {
            engine = try await StreamingASR(
                modelURL: modelURL,
                language: language,
                chunkSize: chunkSize
            )
        } catch {
            continuation.finish(throwing: VoixfulError.engine("model load failed: \(error)"))
            throw VoixfulError.engine("model load failed: \(error)")
        }
    }

    public func consume(_ input: AnalyzerInput) async throws {
        let samples = try converter.convert(input.buffer)
        if utteranceStart == nil {
            utteranceStart = input.bufferStartTime?.seconds ?? 0
        }
        utterance.append(contentsOf: samples)

        // Volatile results are best-effort and throttled; failures are ignored
        // (the authoritative result comes from finalize()).
        guard reportingOptions.contains(.volatileResults), let engine else { return }
        guard utterance.count - lastVolatileCount >= Self.volatileStrideSamples else { return }
        lastVolatileCount = utterance.count
        if let transcript = try? await engine.transcribe(audio: utterance) {
            continuation.yield(makeResult(transcript, isFinal: false))
        }
    }

    public func finalize() async throws {
        guard let engine else {
            continuation.finish()
            return
        }
        do {
            let transcript = try await engine.transcribe(audio: utterance)
            continuation.yield(makeResult(transcript, isFinal: true))
            continuation.finish()
        } catch {
            let wrapped = VoixfulError.engine("transcribe failed: \(error)")
            continuation.finish(throwing: wrapped)
            throw wrapped
        }
    }

    public func cancel() async {
        continuation.finish()
    }

    // MARK: - Result construction

    private func makeResult(_ transcript: Transcript, isFinal: Bool) -> TranscriptionResult {
        let start = utteranceStart ?? 0
        let duration = Double(utterance.count) / 16_000.0

        var text = AttributedString(transcript.text)
        if attributeOptions.contains(.audioTimeRange) {
            // Per-token timing is a Phase-F placeholder in StreamingASR; for now
            // the whole utterance carries one audioTimeRange run.
            text.audioTimeRange = AudioTimeRange(start: start, duration: duration)
        }

        let range = CMTimeRange(
            start: CMTime(seconds: start, preferredTimescale: 1_000),
            duration: CMTime(seconds: duration, preferredTimescale: 1_000)
        )
        return TranscriptionResult(text: text, range: range, isFinal: isFinal)
    }
}
