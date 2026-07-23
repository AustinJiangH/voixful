// LiveFullContextTranscriber — live-microphone partials for BATCH engines.
//
// Full-context backends (Parakeet, Cohere, Granite) have no streaming decode:
// `FullContextTranscriber` buffers everything and transcribes once at
// `finalize`, which is the right shape for file transcription but shows
// nothing during a live mic session. This wrapper gives them live partials
// the way dictation apps do with batch models: re-transcribe the WHOLE
// accumulated utterance on a throttle and report each pass as a volatile
// result, then run one last pass at `finalize` for the final.
//
// Cost model: one full-clip transcribe per refresh. That's only viable
// because these engines are fast (Granite ~0.3 s, Parakeet ~0.07 s for
// clips in the smallest windows) and window-laddered — short live buffers
// hit small windows. Refreshes never overlap (an in-flight pass gates the
// next; audio keeps accumulating meanwhile), so a slow pass degrades to a
// lower refresh rate, never to a queue.

import AVFoundation
import Foundation
import VoixfulEngine
import VoixfulSpeech

public actor LiveFullContextTranscriber<Engine: FullContextEngine>: Transcribing {

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
    private var samplesAtLastRefresh = 0
    private var refreshInFlight = false

    /// New audio required before another refresh pass (16 kHz samples).
    private let refreshStride: Int

    public init(
        modelURL: URL,
        piecesURL: URL,
        locale: Locale = Locale(identifier: "en-US"),
        refreshStrideMS: Int = 700
    ) {
        self.modelURL = modelURL
        self.piecesURL = piecesURL
        self.moduleIdentifier = "com.voixful.transcriber.\(Engine.identifierSuffix).live"
        self.modelName = Engine.displayName
        self.locale = locale
        self.reportingOptions = [.volatileResults]
        self.attributeOptions = []
        self.refreshStride = max(1, refreshStrideMS) * 16
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
        await engine?.prewarm()
    }

    public func consume(_ input: AnalyzerInput) async throws {
        let samples = try converter.convert(input.buffer)
        utterance.append(contentsOf: samples)
        maybeRefresh()
    }

    /// Kick a volatile re-transcribe pass if enough new audio arrived and no
    /// pass is in flight. Detached so the mic feed is never blocked.
    private func maybeRefresh() {
        guard let engine, !refreshInFlight,
              utterance.count - samplesAtLastRefresh >= refreshStride else { return }
        refreshInFlight = true
        samplesAtLastRefresh = utterance.count
        let snapshot = utterance
        Task { [continuation] in
            if let transcript = try? await engine.transcribe(audio: snapshot) {
                continuation.yield(TranscriptionResult(
                    text: AttributedString(transcript.text),
                    range: CMTimeRange(
                        start: .zero,
                        duration: CMTime(seconds: Double(snapshot.count) / 16_000, preferredTimescale: 1_000)),
                    isFinal: false))
            }
            await self.refreshDone()
        }
    }

    private func refreshDone() { refreshInFlight = false }

    public func finalize() async throws {
        guard let engine else { continuation.finish(); return }
        do {
            let transcript = try await engine.transcribe(audio: utterance)
            let duration = Double(utterance.count) / 16_000.0
            continuation.yield(TranscriptionResult(
                text: AttributedString(transcript.text),
                range: CMTimeRange(
                    start: .zero,
                    duration: CMTime(seconds: duration, preferredTimescale: 1_000)),
                isFinal: true))
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
}
