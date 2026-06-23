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
// Two processing modes (DESIGN_ANALYZER_API.md §5):
//   .batch     — strategy (A): accumulate the whole utterance, transcribe it
//                each update. Most accurate (full reflect-padded mel context),
//                but per-update cost grows with utterance length. Best for
//                finite files.
//   .streaming — strategy (B): drive StreamingASR.feed()/finish() incrementally.
//                Fixed cost per chunk, no re-transcription — stays fast for
//                open-ended live capture. Uses the streaming mel path, which
//                trades a little accuracy for constant-time updates.

import Foundation
import CoreMedia
import VoixfulSpeech
import NemotronASR

/// How the Nemotron backend consumes audio. See type-level docs above.
public enum ProcessingMode: Sendable, Equatable {
    case batch
    case streaming
}

public actor NemotronTranscriber: Transcribing {

    // MARK: - Caller-facing surface (synchronous)

    public nonisolated let moduleIdentifier = "com.voixful.transcriber.nemotron"
    public nonisolated let modelName = "Nemotron"
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
    private let mode: ProcessingMode
    private let debug: Bool
    private let converter = AudioInputConverter()

    private var engine: StreamingASR?
    private var utterance: [Float] = []   // batch mode only
    private var pendingStream: [Float] = []   // streaming mode — batched before feed()
    private var samplesSeen = 0           // both modes — drives the time range
    private var utteranceStart: Double?
    private var lastVolatileCount = 0      // batch mode only

    /// Streaming mode feeds the engine in blocks of at least this many samples
    /// (≈1 s @ 16 kHz). The streaming mel path degrades when fed many tiny
    /// buffers (per-call reflect-pad gaps); larger blocks keep it close to the
    /// single-buffer path that transcribes accurately. Still O(1) per chunk.
    private static let streamFeedThreshold = 16_000

    /// Minimum new audio (samples @ 16 kHz) between volatile re-transcriptions
    /// in `.batch` mode. 8000 = 0.5 s; keeps the re-transcribe cost off the
    /// per-buffer hot path. Unused in `.streaming` mode (feed() is already
    /// chunk-paced).
    private static let volatileStrideSamples = 8_000

    // MARK: - Init

    /// Create a transcriber. Model loading is deferred to `prepare()` (driven by
    /// the analyzer) so this initializer stays synchronous like Apple's.
    ///
    /// - Parameter mode: `.batch` (default, accurate, for files) or `.streaming`
    ///   (fixed cost per chunk, for live capture).
    /// - Throws: `VoixfulError.unsupportedLocale` if `locale` is outside the
    ///   backend's `supportedLocales`, or `.engine` for an unsupported chunk size.
    public init(
        modelURL: URL,
        locale: Locale = Locale(identifier: "en-US"),
        chunkSizeMS: Int = 320,
        mode: ProcessingMode = .batch,
        reportingOptions: ReportingOption = [],
        attributeOptions: ResultAttributeOption = [],
        debug: Bool = false
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
        self.mode = mode
        self.debug = debug
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
            let engine = try await StreamingASR(
                modelURL: modelURL,
                language: language,
                chunkSize: chunkSize
            )
            // Streaming mode feeds incrementally, so the engine must be reset +
            // SOS-warmed up front. Batch mode lets transcribe() reset per call.
            if mode == .streaming {
                await engine.start()
            }
            self.engine = engine
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
        samplesSeen += samples.count
        guard let engine else { return }

        switch mode {
        case .batch:
            utterance.append(contentsOf: samples)
            // Volatile results are best-effort and throttled; failures are
            // ignored (the authoritative result comes from finalize()).
            guard reportingOptions.contains(.volatileResults) else { return }
            guard utterance.count - lastVolatileCount >= Self.volatileStrideSamples else { return }
            lastVolatileCount = utterance.count
            if let transcript = try? await engine.transcribe(audio: utterance) {
                continuation.yield(makeResult(text: transcript.text, isFinal: false))
            }

        case .streaming:
            // Batch into ≥~1 s blocks before feeding — the streaming mel path
            // degrades on many tiny buffers. Cost stays O(1) per chunk.
            pendingStream.append(contentsOf: samples)
            guard pendingStream.count >= Self.streamFeedThreshold else { return }
            let block = pendingStream
            pendingStream.removeAll(keepingCapacity: true)
            let partials = try await engine.feed(block)
            if debug {
                logDebug("feed=\(block.count) peak=\(Self.peak(block)) "
                    + "partials=\(partials.count) text='\(partials.last?.text ?? "")'")
            }
            guard reportingOptions.contains(.volatileResults), let last = partials.last else { return }
            continuation.yield(makeResult(text: last.text, isFinal: false))
        }
    }

    public func finalize() async throws {
        guard let engine else {
            continuation.finish()
            return
        }
        do {
            let transcript: Transcript
            switch mode {
            case .batch:
                transcript = try await engine.transcribe(audio: utterance)
            case .streaming:
                // Flush any audio held back below the feed threshold.
                if !pendingStream.isEmpty {
                    _ = try await engine.feed(pendingStream)
                    pendingStream.removeAll(keepingCapacity: true)
                }
                transcript = try await engine.finish()
            }
            if debug { logDebug("finalize text='\(transcript.text)'") }
            continuation.yield(makeResult(text: transcript.text, isFinal: true))
            continuation.finish()
        } catch {
            let wrapped = VoixfulError.engine("finalize failed: \(error)")
            continuation.finish(throwing: wrapped)
            throw wrapped
        }
    }

    public func cancel() async {
        continuation.finish()
    }

    // MARK: - Result construction

    private func makeResult(text rawText: String, isFinal: Bool) -> TranscriptionResult {
        let start = utteranceStart ?? 0
        let duration = Double(samplesSeen) / 16_000.0

        var text = AttributedString(rawText)
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

    // MARK: - Debug

    private nonisolated func logDebug(_ message: String) {
        FileHandle.standardError.write(Data("[dbg] \(message)\n".utf8))
    }

    private static func peak(_ samples: [Float]) -> String {
        var p: Float = 0
        for s in samples { p = max(p, abs(s)) }
        return String(format: "%.4f", p)
    }
}
