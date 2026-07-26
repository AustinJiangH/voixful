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
    /// A HEAD-COMMIT pass is running. It freezes text into `committedText` and
    /// drops those samples from `utterance`, so it MUST be awaited (abandoning it
    /// would lose audio / committed text). `finalize` waits this out.
    private var commitInFlight = false
    /// A plain VOLATILE refresh pass is running. It only yields a volatile
    /// partial and mutates no committed state, so it is safe to abandon — the
    /// final pass supersedes it. `finalize` drops it immediately.
    private var refreshInFlight = false
    /// Bumped to invalidate an in-flight/unapplied plain refresh: a refresh Task
    /// applies its `yield` (and clears `refreshInFlight`) only while its captured
    /// generation still matches. `finalize` bumps it so the volatile pass it
    /// abandons can't yield after the final.
    private var refreshGeneration = 0
    /// Set by `cancel()`. Once true, `maybeRefresh` starts no NEW pass and the
    /// results stream is finished — so a cancelled live job promptly frees the
    /// shared (resident) engine for the superseding job.
    private var cancelled = false
    /// Set by `finalize()`. Stops a post-commit `maybeRefresh` (fired from
    /// `commitHead` after a head-commit that `finalize` awaited) from launching a
    /// fresh volatile pass that would race the final — the tail is about to be
    /// transcribed by the final pass anyway.
    private var finalizing = false
    /// The single in-flight detached refresh/commit pass (only one runs at a
    /// time — `refreshInFlight` gates it). Held so `cancel()` can cancel it,
    /// which trips `Task.isCancelled` inside the engine's decode loop so even the
    /// current pass bails early instead of running to completion.
    private var refreshTask: Task<Void, Never>?
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
    /// Live-buffer size that triggers a head commit (16 kHz samples) — the
    /// streaming analog of `LongForm` segmentation. Clamped in `prepare()` to
    /// below the engine's fixed window so each pass stays within one window and
    /// the head is committed (never dropped) as the utterance grows past it.
    private var commitThreshold: Int

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

    /// Wrap an ALREADY-loaded, prewarmed engine (residency — see
    /// `ResidentFullContextModel`). `prepare()` early-returns on a pre-set engine,
    /// so this makes the per-utterance transcriber a near no-op to spin up while
    /// reusing the shared warm model. Per-utterance buffers are fresh by virtue of
    /// being a new instance. The `piecesURL` is unused here (the engine is loaded)
    /// but kept in the signature for parity with the `modelURL:` init.
    package init(
        engine: Engine,
        piecesURL: URL,
        locale: Locale = Locale(identifier: "en-US"),
        refreshStrideMS: Int = 700,
        commitThresholdSeconds: Int = 24
    ) {
        self.modelURL = piecesURL   // unused: prepare() early-returns on the injected engine
        self.piecesURL = piecesURL
        self.moduleIdentifier = "com.voixful.transcriber.\(Engine.identifierSuffix).live"
        self.modelName = Engine.displayName
        self.locale = locale
        self.reportingOptions = [.volatileResults]
        self.attributeOptions = []
        self.refreshStride = max(1, refreshStrideMS) * 16
        // prepare() normally clamps commitThreshold to the engine's window, but it
        // early-returns on a pre-set engine — so do the clamp here from the
        // (nonisolated) window instead.
        let requested = max(8, commitThresholdSeconds) * 16_000
        let safeWindow = Int(Double(engine.maxAudioSamples) * 0.85)
        self.commitThreshold = min(requested, safeWindow)
        let (stream, continuation) = AsyncThrowingStream<TranscriptionResult, Error>.makeStream()
        self.results = stream
        self.continuation = continuation
        self.engine = engine
    }

    public func prepare() async throws {
        guard engine == nil else { return }
        let created: Engine
        do {
            created = try await Engine(modelURL: modelURL, piecesURL: piecesURL)
        } catch {
            throw VoixfulError.engine("could not load \(Engine.displayName) engine: \(error)")
        }
        engine = created

        // Commit the head BEFORE the growing buffer reaches the engine's fixed
        // window (Parakeet: 10 s): the head is transcribed and frozen, the tail
        // keeps growing — no audio is dropped and no pass exceeds one window.
        // 85% leaves headroom for a refresh stride and the split point. Clamp
        // strictly DOWN to safeWindow (never up), and never above the caller's
        // requested threshold.
        let safeWindow = Int(Double(created.maxAudioSamples) * 0.85)
        commitThreshold = min(commitThreshold, safeWindow)

        await created.prewarm()
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
        guard !cancelled, !finalizing, let engine, !commitInFlight, !refreshInFlight,
              utterance.count - samplesAtLastRefresh >= refreshStride else { return }
        if !producedFirstResult {
            continuation.yield(TranscriptionResult(
                text: AttributedString("[ preparing \(modelName) — first pass can take a while ]"),
                range: CMTimeRange(start: .zero, duration: .zero),
                isFinal: false))
        }
        samplesAtLastRefresh = utterance.count

        if utterance.count >= commitThreshold {
            commitInFlight = true
            let splitAt = LongForm.quietestSplit(
                utterance, from: commitThreshold / 2, to: commitThreshold)
            let head = Array(utterance[..<splitAt])
            refreshTask = Task {
                do {
                    // LongForm keeps the head safe even if it ever exceeds the
                    // window; on a transcribe error we must NOT drop the samples.
                    let text = try await LongForm.transcribe(audio: head, with: engine)
                    self.commitHead(text: text, samples: splitAt)
                } catch {
                    self.commitFailed()
                }
            }
            return
        }

        // Plain volatile refresh. Tag it with the current generation so a
        // `finalize` that bumps the generation while this pass is in flight makes
        // it drop its (about-to-be-superseded) result instead of yielding it.
        let gen = refreshGeneration
        refreshInFlight = true
        let snapshot = utterance
        let prefix = committedText
        refreshTask = Task { [continuation] in
            let transcript = try? await engine.transcribe(audio: snapshot)
            if gen == self.refreshGeneration, let transcript {
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
            self.refreshDone(gen: gen)
        }
    }

    /// A plain refresh Task finished. Only clear `refreshInFlight` if this pass is
    /// still current — a `finalize` that abandoned it already cleared the flag and
    /// moved the generation on, and clearing it here for a stale pass is a no-op
    /// we skip to keep the intent explicit.
    private func refreshDone(gen: Int) {
        producedFirstResult = true
        guard gen == refreshGeneration else { return }
        refreshInFlight = false
    }

    private func commitHead(text: String, samples: Int) {
        // A completed commit means the model is warm — stop re-showing the
        // "[ preparing… ]" banner in a commit-dominated session (large models /
        // long dictation, where plain refreshes may never land between commits).
        producedFirstResult = true
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.isEmpty {
            committedText = committedText.isEmpty ? t : committedText + " " + t
        }
        utterance.removeFirst(min(samples, utterance.count))
        samplesAtLastRefresh = 0
        commitInFlight = false
        maybeRefresh()  // tail may already be past the refresh stride
    }

    /// A head-commit transcribe failed (e.g. transient specialization/OOM).
    /// Keep the head samples in `utterance` so the next pass retries — a failed
    /// commit must never silently drop audio. Reset the refresh accounting so
    /// the retry fires without waiting a whole extra stride.
    private func commitFailed() {
        commitInFlight = false
        samplesAtLastRefresh = 0
    }

    public func finalize() async throws {
        // Re-entry, or finalize-after-cancel: an in-flight finalize (or cancel())
        // already owns the terminal. Return WITHOUT finishing — a second call must
        // not cut off the first finalize's `isFinal`, nor (in the resident/
        // supersede path) run a second heavy `LongForm.transcribe` that re-occupies
        // the shared engine and delays a superseding job. cancel() already
        // finished the stream, so there's nothing left to do.
        guard !finalizing, !cancelled else { return }
        guard let engine else { continuation.finish(); return }
        do {
            finalizing = true
            // A plain volatile refresh is about to be superseded by the final
            // pass, so don't wait it out: bump the generation (its result, if it
            // ever lands, is now stale and won't yield) and — when no head-commit
            // is in flight — drop our in-flight flag and cancel the detached pass
            // so it stops burning the ANE and frees it for the final sooner. A
            // head-commit, by contrast, mutates committedText/utterance and must
            // never be cancelled: leave `refreshTask` alone while one is running.
            refreshGeneration &+= 1
            if !commitInFlight {
                refreshInFlight = false
                refreshTask?.cancel()
            }
            // Still wait out an in-flight HEAD COMMIT so the tail buffer and
            // committedText are settled before the final pass (it is not
            // abandonable — see above).
            while commitInFlight {
                try await Task.sleep(nanoseconds: 20_000_000)
            }
            // Segment defensively: the tail is normally below the window after
            // commits, but a final burst with no pause could push it over.
            let tail = try await LongForm.transcribe(audio: utterance, with: engine)
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

    /// Prompt, cooperative cancel. Sets the stop flag so no NEW refresh pass is
    /// started, cancels the single in-flight detached pass (which trips
    /// `Task.isCancelled` in the engine's decode loop so even that pass bails
    /// early), and finishes the results stream — WITHOUT awaiting the in-flight
    /// pass. So the shared (resident) engine is freed for the next job with at
    /// most one already-dispatched Core AI call left to drain; no final result is
    /// emitted for the cancelled job.
    public func cancel() async {
        cancelled = true
        refreshTask?.cancel()
        refreshTask = nil
        continuation.finish()
    }
}
