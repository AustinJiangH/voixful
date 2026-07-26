// Deterministic tests for the finalize latency fix in LiveFullContextTranscriber
// — NO real model. On key-release, `finalize()` must abandon an in-flight PLAIN
// volatile refresh immediately (the final pass supersedes it) yet still await an
// in-flight HEAD COMMIT (it mutates committed state / would drop audio).
//
// A gated, word-decoding mock (SlowCancellableMock's blocking + LongFormMock's
// synthetic-word decode) makes the timing deterministic without wall-clock
// sleeps: each `transcribe` blocks — staying observably in flight — until the
// test opens the gate (then it decodes) or the Task is cancelled (then it bails,
// like Parakeet's `Task.checkCancellation()` decode loop). Tests drive finalize
// concurrently and open the gate only after observing the abandon/await, so a
// regression changes WHICH passes bail vs complete — not just how long they take.

import AVFoundation
import XCTest
import VoixfulEngine
import VoixfulFullContext
@testable import VoixfulSpeech

// MARK: - Gate control (poll-based, cross-actor via a shared global)

/// Shared control/observation for `GatedWordMock`: watch a pass go in-flight,
/// open the gate to let passes complete, and count how many BAILED (cancelled)
/// vs ran to COMPLETION. Poll-based like the resident tests' `SlowMockControl`.
private final class GateControl: @unchecked Sendable {
    private let lock = NSLock()
    private var _open = false
    private var _inFlight = 0
    private var _bailed = 0
    private var _completed = 0

    var isOpen: Bool { lock.lock(); defer { lock.unlock() }; return _open }
    var inFlight: Int { lock.lock(); defer { lock.unlock() }; return _inFlight }
    var bailed: Int { lock.lock(); defer { lock.unlock() }; return _bailed }
    var completed: Int { lock.lock(); defer { lock.unlock() }; return _completed }

    /// Let any in-flight (and future) pass decode instead of blocking.
    func open() { lock.lock(); _open = true; lock.unlock() }
    func enter() { lock.lock(); _inFlight += 1; lock.unlock() }
    func exitBailed() { lock.lock(); _inFlight -= 1; _bailed += 1; lock.unlock() }
    func exitCompleted() { lock.lock(); _inFlight -= 1; _completed += 1; lock.unlock() }
    func reset() {
        lock.lock(); _open = false; _inFlight = 0; _bailed = 0; _completed = 0; lock.unlock()
    }
}

/// Global control shared by every `GatedWordMock` (the transcriber constructs its
/// engine via `Engine(modelURL:piecesURL:)`, so the control is reached the same
/// way the other mock tests reach theirs). Reset at the top of each test.
private let gate = GateControl()

/// A `FullContextEngine` whose `transcribe` blocks — staying observably in
/// flight — until the gate opens (then it decodes each maximal run of a distinct
/// non-zero value as one word "w<v>", like LongFormMockTests) or the Task is
/// cancelled (then it bails via `CancellationError`, like the real decode loop).
private actor GatedWordMock: FullContextEngine {
    static let displayName = "GatedWord"
    static let identifierSuffix = "gatedword"
    static var supportedLocales: [Locale] { [Locale(identifier: "en-US")] }
    // 1 s window — a modest live buffer stays inside one window (plain refresh),
    // so tests choose the commit-vs-refresh branch via buffer size, not the mock.
    nonisolated let maxAudioSamples = 16_000

    init(modelURL: URL, piecesURL: URL) async throws {}
    func prewarm() {}

    func transcribe(audio: [Float]) async throws -> Transcript {
        gate.enter()
        // Block until the gate opens (decode) or the Task is cancelled (bail) —
        // no wall-clock cap; XCTest's own timeout is the hang guard.
        while true {
            if Task.isCancelled { gate.exitBailed(); throw CancellationError() }
            if gate.isOpen { gate.exitCompleted(); return decode(audio) }
            try? await Task.sleep(for: .milliseconds(2))
        }
    }

    private func decode(_ audio: [Float]) -> Transcript {
        var tokens: [String] = []
        var prev: Float = 0
        for v in audio {
            if v != 0, v != prev { tokens.append("w\(Int(v))") }
            prev = v
        }
        return Transcript(text: tokens.joined(separator: " "), segments: [], language: nil)
    }
}

final class LiveFinalizeLatencyTests: XCTestCase {

    private let nul = URL(fileURLWithPath: "/dev/null")

    // MARK: helpers

    private func makeLive(refreshStrideMS: Int = 100, commitThresholdSeconds: Int = 24)
        -> LiveFullContextTranscriber<GatedWordMock> {
        LiveFullContextTranscriber<GatedWordMock>(
            modelURL: nul, piecesURL: nul,
            refreshStrideMS: refreshStrideMS, commitThresholdSeconds: commitThresholdSeconds)
    }

    /// Feed `audio` as a single mic buffer (content matters — words are decoded).
    private func feed(_ audio: [Float], into t: any Transcribing) async throws {
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
                                channels: 1, interleaved: false)!
        let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(audio.count))!
        buf.frameLength = AVAudioFrameCount(audio.count)
        audio.withUnsafeBufferPointer { src in
            buf.floatChannelData![0].update(from: src.baseAddress!, count: audio.count)
        }
        try await t.consume(AnalyzerInput(buffer: buf))
    }

    /// `count` words of distinct value 1…N (1600 samples each) separated by 3200
    /// samples of silence — the gaps are wider than a word so segment cuts land
    /// in silence between words.
    private func makeWords(_ count: Int) -> [Float] {
        var audio: [Float] = []
        for k in 1...count {
            audio.append(contentsOf: [Float](repeating: Float(k), count: 1_600))
            audio.append(contentsOf: [Float](repeating: 0, count: 3_200))
        }
        return audio
    }

    private func collect(_ t: any Transcribing) -> Task<[TranscriptionResult], Never> {
        let stream = t.results
        return Task {
            var out: [TranscriptionResult] = []
            do { for try await r in stream { out.append(r) } } catch {}
            return out
        }
    }

    private func poll(_ timeout: Duration = .seconds(5), until: @Sendable () -> Bool) async {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        while clock.now < deadline {
            if until() { return }
            try? await Task.sleep(for: .milliseconds(3))
        }
    }

    private func makeInput(frames: Int, value: Float = 0.01) -> AnalyzerInput {
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
                                channels: 1, interleaved: false)!
        let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(frames))!
        buf.frameLength = AVAudioFrameCount(frames)
        buf.floatChannelData![0].update(repeating: value, count: frames)
        return AnalyzerInput(buffer: buf, bufferStartTime: .zero)
    }

    // MARK: - Item 2: finalize abandons an in-flight plain refresh

    // Release mid PLAIN refresh: finalize must NOT wait out the volatile pass. It
    // cancels it (bail) and proceeds straight to the final pass — deterministic
    // because the abandoned pass never completes on its own (gate stays closed
    // until the test opens it, and only AFTER the bail, for the final pass). Under
    // the old `while refreshInFlight { sleep }`, finalize would hang until the
    // stuck pass completed and then that pass would yield its stale volatile.
    func testFinalizeAbandonsInFlightPlainRefresh() async throws {
        gate.reset()
        let live = makeLive()
        try await live.prepare()
        let collector = collect(live)

        // 3200 samples (value 5) → one word "w5": ≥ the 100 ms stride (1600) but
        // far below the commit threshold → the plain-refresh branch.
        try await feed([Float](repeating: 5, count: 3_200), into: live)
        await poll { gate.inFlight == 1 }
        XCTAssertEqual(gate.inFlight, 1, "a plain refresh pass should be in flight")

        // finalize runs concurrently so we can open the gate for the FINAL pass
        // only after the volatile pass has been abandoned.
        let fin = Task { try await live.finalize() }
        await poll { gate.bailed == 1 }
        XCTAssertEqual(gate.bailed, 1, "finalize must abandon (cancel) the in-flight plain refresh")
        XCTAssertEqual(gate.completed, 0, "the abandoned refresh must not run to completion")

        gate.open()                 // now let the final pass decode
        try await fin.value

        let results = await collector.value
        let finals = results.filter { $0.isFinal }
        XCTAssertEqual(finals.count, 1, "exactly one final result")
        XCTAssertEqual(String(finals.first!.text.characters), "w5", "final content unchanged")
        // Generation guard: the abandoned refresh never yielded its volatile.
        let strayVolatile = results.contains { !$0.isFinal && String($0.text.characters) == "w5" }
        XCTAssertFalse(strayVolatile, "abandoned refresh must not yield a volatile after finalize began")
    }

    // Release mid HEAD COMMIT: finalize must WAIT for it (it freezes committedText
    // and drops those samples — abandoning it would lose audio). The commit runs
    // to completion (never bails), the committed head words are present in the
    // final, and no words are dropped.
    func testFinalizeAwaitsInFlightHeadCommit() async throws {
        gate.reset()
        let live = makeLive()
        try await live.prepare()
        let collector = collect(live)

        // 4 words + gaps = 19 200 samples in one buffer → past the 13 600 commit
        // threshold (0.85 × 16 kHz) → the FIRST pass is a head commit.
        try await feed(makeWords(4), into: live)
        await poll { gate.inFlight == 1 }
        XCTAssertEqual(gate.inFlight, 1, "a head-commit pass should be in flight")

        let fin = Task { try await live.finalize() }
        // Give finalize a beat to reach its `while commitInFlight` wait; it must
        // NOT have cancelled the commit.
        try? await Task.sleep(for: .milliseconds(30))
        XCTAssertEqual(gate.bailed, 0, "finalize must not abandon a head-commit")

        gate.open()                 // commit completes, then the final tail pass
        try await fin.value

        XCTAssertEqual(gate.bailed, 0, "the head-commit must run to completion, not bail")
        let results = await collector.value
        let finals = results.filter { $0.isFinal }
        XCTAssertEqual(finals.count, 1, "exactly one final result")
        let text = String(finals.first!.text.characters)
        let tokens = Set(text.split(separator: " ").map(String.init))
        let missing = (1...4).map { "w\($0)" }.filter { !tokens.contains($0) }
        XCTAssertTrue(missing.isEmpty, "no audio dropped — committed head + tail present: \(text)")
    }

    // cancel() during an in-flight pass still finishes cleanly with NO final
    // (unchanged hard-abort semantics — cancel never runs a final pass).
    func testCancelDuringInFlightPassEmitsNoFinal() async throws {
        gate.reset()
        let live = makeLive()
        try await live.prepare()
        let collector = collect(live)

        try await feed([Float](repeating: 5, count: 3_200), into: live)
        await poll { gate.inFlight == 1 }

        await live.cancel()
        await poll { gate.bailed == 1 }
        XCTAssertEqual(gate.bailed, 1, "cancel must bail the in-flight pass")

        let results = await collector.value       // stream finished by cancel()
        XCTAssertFalse(results.contains { $0.isFinal }, "a cancelled job emits no final result")
    }

    // finalize() AFTER cancel() (reachable in the resident supersede path: a new
    // vend cancels the prior, then the app's end() still calls finalize()) must
    // run NO final pass — it would re-occupy the shared engine and delay the
    // superseding job. Gate is OPEN, so a leaked final pass would `complete`.
    func testFinalizeAfterCancelRunsNoFinalPass() async throws {
        gate.reset()
        gate.open()                     // any pass that runs would complete immediately
        let live = makeLive()
        try await live.prepare()
        let collector = collect(live)

        // 800 samples < the 1600-sample (100 ms) stride → no pass fires on consume.
        try await feed([Float](repeating: 5, count: 800), into: live)
        XCTAssertEqual(gate.completed, 0, "no pass should have run yet")

        await live.cancel()             // hard abort — finishes the stream
        try await live.finalize()       // must be a no-op (guard on cancelled)

        XCTAssertEqual(gate.completed, 0, "finalize after cancel must run no final pass")
        XCTAssertEqual(gate.bailed, 0)
        let results = await collector.value
        XCTAssertFalse(results.contains { $0.isFinal }, "no final result after cancel")
    }

    // MARK: - Item 1: refreshStride is threaded through the resident load path

    // A non-default refreshStrideMS must reach the vended resident transcriber. A
    // 2000-sample buffer exceeds the 100 ms stride (1600) but is far below the
    // 700 ms default (11 200): a pass fires ONLY if the non-default stride was
    // threaded ResidentContext → ResidentFullContextModel → FullContextResidentBackend
    // → LiveFullContextTranscriber. With the old default it would not fire.
    func testResidentRefreshStrideReachesTranscriber() async throws {
        gate.reset()
        let model = ResidentFullContextModel<GatedWordMock>(
            modelURL: nul, piecesURL: nul,
            refreshStrideMS: 100, commitThresholdSeconds: 24,
            offload: .pinned, clock: ContinuousClock())
        let t = try await model.transcriber(locale: nil)
        try await t.prepare()

        try await t.consume(makeInput(frames: 2_000))
        await poll { gate.inFlight == 1 }
        XCTAssertEqual(gate.inFlight, 1,
                       "non-default refreshStride (100 ms) must reach the resident transcriber")

        await t.cancel()            // release the pin so unload can offload
        await poll { gate.bailed == 1 }
        await model.unload()
    }

    // Control: at the 700 ms DEFAULT stride the same 2000-sample buffer is below
    // the 11 200-sample stride, so NO pass fires — the discriminator for the test
    // above (proves it isn't firing for some unrelated reason).
    func testResidentDefaultStrideDoesNotFireOnSmallBuffer() async throws {
        gate.reset()
        let model = ResidentFullContextModel<GatedWordMock>(
            modelURL: nul, piecesURL: nul, offload: .pinned, clock: ContinuousClock())
        let t = try await model.transcriber(locale: nil)
        try await t.prepare()

        try await t.consume(makeInput(frames: 2_000))
        // Give any errant pass a chance to dispatch.
        try? await Task.sleep(for: .milliseconds(50))
        XCTAssertEqual(gate.inFlight, 0,
                       "at the 700 ms default a 2000-sample buffer must not trigger a pass")

        await t.cancel()
        await model.unload()
    }
}
