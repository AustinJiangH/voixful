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
    /// Transcript of audio already COMMITTED and dropped from `utterance`.
    /// Long dictation would otherwise outgrow the engine's largest window
    /// (which silently truncates) and make every refresh pass slower: once
    /// the live buffer passes `commitThreshold`, the head is split at the
    /// quietest gap, transcribed once, its text frozen here, and only the
    /// tail keeps re-transcribing — refreshes get fast again after each
    /// commit and no audio is ever dropped.
    private var committedText = ""

    /// New audio required before another refresh pass (16 kHz samples).
    private let refreshStride: Int
    /// Live-buffer size that triggers a head commit (16 kHz samples).
    private let commitThreshold: Int

    public init(
        modelURL: URL,
        piecesURL: URL,
        locale: Locale = Locale(identifier: "en-US"),
        refreshStrideMS: Int = 700,
        commitThresholdSeconds: Int = 24
    ) {
        self.modelURL = modelURL
        self.piecesURL = piecesURL
        self.moduleIdentifier = "com.voixful.transcriber.\(Engine.identifierSuffix).live"
        self.modelName = Engine.displayName
        self.locale = locale
        self.reportingOptions = [.volatileResults]
        self.attributeOptions = []
        self.refreshStride = max(1, refreshStrideMS) * 16
        self.commitThreshold = max(8, commitThresholdSeconds) * 16_000
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

    /// True once any transcribe pass has produced a result. Until then the
    /// UI shows a warming notice — a fresh model's first pass can spend tens
    /// of seconds in one-time graph specialization, which otherwise looks
    /// like a dead session (meter moves, no words).
    private var producedFirstResult = false

    public func consume(_ input: AnalyzerInput) async throws {
        let samples = try converter.convert(input.buffer)
        utterance.append(contentsOf: samples)
        maybeRefresh()
    }

    /// Kick a volatile re-transcribe pass if enough new audio arrived and no
    /// pass is in flight. When the live buffer has outgrown the commit
    /// threshold, this pass doubles as a head commit: transcribe up to the
    /// quietest gap, freeze that text, drop those samples. Detached so the
    /// mic feed is never blocked.
    private func maybeRefresh() {
        guard let engine, !refreshInFlight,
              utterance.count - samplesAtLastRefresh >= refreshStride else { return }
        if !producedFirstResult {
            continuation.yield(TranscriptionResult(
                text: AttributedString("[ preparing \(modelName) — first pass can take a while ]"),
                range: CMTimeRange(start: .zero, duration: .zero),
                isFinal: false))
        }
        refreshInFlight = true
        samplesAtLastRefresh = utterance.count

        if utterance.count >= commitThreshold {
            let splitAt = Self.quietestSplit(
                utterance, from: commitThreshold / 2, to: commitThreshold)
            let head = Array(utterance[..<splitAt])
            Task {
                let text = (try? await engine.transcribe(audio: head))?.text ?? ""
                await self.commitHead(text: text, samples: splitAt)
            }
            return
        }

        let snapshot = utterance
        let prefix = committedText
        Task { [continuation] in
            if let transcript = try? await engine.transcribe(audio: snapshot) {
                let joined = prefix.isEmpty
                    ? transcript.text
                    : prefix + " " + transcript.text
                continuation.yield(TranscriptionResult(
                    text: AttributedString(joined),
                    range: CMTimeRange(
                        start: .zero,
                        duration: CMTime(seconds: Double(snapshot.count) / 16_000, preferredTimescale: 1_000)),
                    isFinal: false))
            }
            await self.refreshDone()
        }
    }

    private func refreshDone() {
        refreshInFlight = false
        producedFirstResult = true
    }

    private func commitHead(text: String, samples: Int) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.isEmpty {
            committedText = committedText.isEmpty ? t : committedText + " " + t
        }
        utterance.removeFirst(min(samples, utterance.count))
        samplesAtLastRefresh = 0
        refreshInFlight = false
        maybeRefresh()  // tail may already be past the refresh stride
    }

    /// Index of the quietest 200 ms in `[from, to)` — a cheap pause detector
    /// so head commits split between words, not through them. Returns the
    /// middle of the quietest window.
    private static func quietestSplit(_ audio: [Float], from: Int, to: Int) -> Int {
        let win = 3_200  // 200 ms at 16 kHz
        let hi = min(to, audio.count) - win
        guard hi > from else { return min(to, audio.count) }
        var best = from
        var bestEnergy = Float.greatestFiniteMagnitude
        var i = from
        while i < hi {
            var e: Float = 0
            for j in i..<(i + win) { e += audio[j] * audio[j] }
            if e < bestEnergy {
                bestEnergy = e
                best = i
            }
            i += win / 2
        }
        return best + win / 2
    }

    public func finalize() async throws {
        guard let engine else { continuation.finish(); return }
        do {
            // Wait out any in-flight refresh/commit so the tail buffer and
            // committedText are settled before the final pass.
            while refreshInFlight {
                try await Task.sleep(nanoseconds: 20_000_000)
            }
            let tail = try await engine.transcribe(audio: utterance).text
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let full = committedText.isEmpty
                ? tail
                : (tail.isEmpty ? committedText : committedText + " " + tail)
            let duration = Double(utterance.count) / 16_000.0
            continuation.yield(TranscriptionResult(
                text: AttributedString(full),
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
