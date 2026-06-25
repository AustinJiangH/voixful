// VoixfulParakeet — Parakeet TDT 0.6B as a `Transcribing` backend behind the
// Voixful façade (DESIGN_ANALYZER_API.md §4: "one backend behind a shared
// façade"). Full-context: buffers all input via `consume`, transcribes once at
// `finalize`. Keeps the engine (`ParakeetEngine`) and Core AI types internal.

import AVFoundation
import Foundation
import VoixfulEngine
import VoixfulSpeech

public actor ParakeetTranscriber: Transcribing {

    // MARK: Transcribing / VoixfulSpeechModule
    public nonisolated let moduleIdentifier = "voixful.parakeet"
    public nonisolated let modelName = "Parakeet TDT 0.6B"
    public nonisolated let locale: Locale
    public nonisolated let reportingOptions: ReportingOption
    public nonisolated let attributeOptions: ResultAttributeOption
    public nonisolated let results: AsyncThrowingStream<TranscriptionResult, Error>

    /// Parakeet TDT v3 is multilingual; the locale is a display/metadata hint —
    /// the model auto-detects language. en-US is the default.
    public static let supportedLocales: [Locale] = [Locale(identifier: "en-US")]

    private nonisolated let continuation: AsyncThrowingStream<TranscriptionResult, Error>.Continuation
    private let modelURL: URL
    private let piecesURL: URL
    private let converter = AudioInputConverter()
    private var engine: ParakeetEngine?
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
            engine = try await ParakeetEngine(modelURL: modelURL, piecesURL: piecesURL)
        } catch {
            throw VoixfulError.engine("could not load Parakeet engine: \(error)")
        }
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
            let text = try await engine.transcribe(audio: utterance)
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
