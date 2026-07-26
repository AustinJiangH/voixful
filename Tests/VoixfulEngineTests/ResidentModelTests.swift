// Deterministic residency tests — NO real model. A mock FullContextEngine with a
// shared init counter (like LongFormMockTests) verifies the ResidentModel
// lifecycle: engine reuse, idle offload + transparent reload, activity reset,
// deferred offload while a transcriber is outstanding, coalesced concurrent
// reloads, pin release on finalize/cancel AND via deinit, and warm-on-load.
//
// The idle timer runs on an injected `ManualClock` so time advances only when
// the test says so — no wall-clock flakiness.

import XCTest
import AVFoundation
import VoixfulEngine
import VoixfulFullContext
import VoixfulKit
@testable import VoixfulSpeech

// Compile-only: pin the public entry point's shape (never called — no real model
// on CI). If `ResidentModel.load(model:locale:backend:offload:)` ever drifts from
// the design, this stops compiling.
@available(*, unavailable)
private func _residentLoadEntryPointShape() async throws {
    _ = try await loadResidentModel(
        model: URL(fileURLWithPath: "/dev/null"),
        locale: Locale(identifier: "en-US"),
        backend: nil,
        offload: .default)
}

// MARK: - Manual clock (time advances only on advance())

private final class ManualClock: Clock, @unchecked Sendable {
    struct Instant: InstantProtocol {
        var offset: Swift.Duration
        func advanced(by duration: Swift.Duration) -> Instant { Instant(offset: offset + duration) }
        func duration(to other: Instant) -> Swift.Duration { other.offset - offset }
        static func < (lhs: Instant, rhs: Instant) -> Bool { lhs.offset < rhs.offset }
    }

    private struct Sleeper { let deadline: Instant; let cont: CheckedContinuation<Void, Error> }

    private let lock = NSLock()
    private var current = Instant(offset: .zero)
    private var pending: [Int: Sleeper] = [:]
    private var nextID = 0

    var now: Instant { lock.lock(); defer { lock.unlock() }; return current }
    var minimumResolution: Swift.Duration { .zero }

    /// Count of LIVE (uncancelled) sleepers — cancelled ones are removed
    /// promptly, so a test can wait until the resident model's idle timer has
    /// actually registered before advancing (removing the scheduling race).
    var liveSleepers: Int { lock.lock(); defer { lock.unlock() }; return pending.count }

    private func makeID() -> Int { lock.lock(); defer { lock.unlock() }; let id = nextID; nextID += 1; return id }

    func sleep(until deadline: Instant, tolerance: Swift.Duration?) async throws {
        let id = makeID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                lock.lock()
                if Task.isCancelled { lock.unlock(); cont.resume(throwing: CancellationError()); return }
                if current >= deadline { lock.unlock(); cont.resume(); return }
                pending[id] = Sleeper(deadline: deadline, cont: cont)
                lock.unlock()
            }
        } onCancel: {
            lock.lock(); let s = pending.removeValue(forKey: id); lock.unlock()
            s?.cont.resume(throwing: CancellationError())
        }
    }

    func advance(by duration: Swift.Duration) {
        lock.lock()
        current = current.advanced(by: duration)
        let due = pending.filter { $0.value.deadline <= current }
        for k in due.keys { pending.removeValue(forKey: k) }
        lock.unlock()
        for (_, s) in due { s.cont.resume() }
    }
}

// MARK: - Counting mock engine

private final class InitCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var _count = 0
    func bump() { lock.lock(); _count += 1; lock.unlock() }
    var count: Int { lock.lock(); defer { lock.unlock() }; return _count }
    func reset() { lock.lock(); _count = 0; lock.unlock() }
}
private let initCounter = InitCounter()

private actor CountingMock: FullContextEngine {
    static let displayName = "CountMock"
    static let identifierSuffix = "countmock"
    static var supportedLocales: [Locale] { [Locale(identifier: "en-US")] }
    nonisolated let maxAudioSamples = 16_000

    init(modelURL: URL, piecesURL: URL) async throws { initCounter.bump() }
    func prewarm() {}
    func transcribe(audio: [Float]) async throws -> Transcript {
        Transcript(text: "", segments: [], language: nil)
    }
}

// MARK: - Slow, cancellable mock engine

/// Shared control/observation for `SlowCancellableMock`, so a test can watch a
/// pass go in-flight, open a gate to let it complete, and confirm whether the
/// pass BAILED on cancellation (promptness) or ran to completion. Poll-based
/// like `InitCounter` to avoid cross-actor plumbing.
private final class SlowMockControl: @unchecked Sendable {
    private let lock = NSLock()
    private var _inFlight = 0
    private var _bailed = 0
    private var _completed = 0
    private var _open = false

    var inFlight: Int { lock.lock(); defer { lock.unlock() }; return _inFlight }
    var bailed: Int { lock.lock(); defer { lock.unlock() }; return _bailed }
    var completed: Int { lock.lock(); defer { lock.unlock() }; return _completed }
    var isOpen: Bool { lock.lock(); defer { lock.unlock() }; return _open }

    /// Let any in-flight (and future) pass complete instead of blocking.
    func open() { lock.lock(); _open = true; lock.unlock() }
    func enter() { lock.lock(); _inFlight += 1; lock.unlock() }
    func exitBailed() { lock.lock(); _inFlight -= 1; _bailed += 1; lock.unlock() }
    func exitCompleted() { lock.lock(); _inFlight -= 1; _completed += 1; lock.unlock() }

    func reset() {
        lock.lock(); _inFlight = 0; _bailed = 0; _completed = 0; _open = false; lock.unlock()
    }
}

/// Global control shared by every `SlowCancellableMock` (the resident backend
/// constructs its engine via `Engine(modelURL:piecesURL:)`, so the control is
/// reached the same way `initCounter` is). Reset at the top of each slow test.
private let slowControl = SlowMockControl()

/// A `FullContextEngine` whose `transcribe` blocks (staying observably
/// in-flight) until either the control's gate opens — then it returns a
/// non-empty transcript — or the calling Task is cancelled — then it bails
/// early via `CancellationError`, exactly like the real Parakeet decode loop's
/// `Task.checkCancellation()`. This makes cancel promptness deterministic
/// without a real model.
private actor SlowCancellableMock: FullContextEngine {
    static let displayName = "SlowMock"
    static let identifierSuffix = "slowmock"
    static var supportedLocales: [Locale] { [Locale(identifier: "en-US")] }
    nonisolated let maxAudioSamples = 16_000

    init(modelURL: URL, piecesURL: URL) async throws { initCounter.bump() }
    func prewarm() {}

    func transcribe(audio: [Float]) async throws -> Transcript {
        slowControl.enter()
        // Bounded spin so a wedged test can't hang forever (~4 s cap).
        for _ in 0..<2000 {
            if Task.isCancelled { slowControl.exitBailed(); throw CancellationError() }
            if slowControl.isOpen {
                slowControl.exitCompleted()
                return Transcript(text: "complete transcript", segments: [], language: nil)
            }
            try? await Task.sleep(for: .milliseconds(2))
        }
        slowControl.exitCompleted()
        return Transcript(text: "complete transcript", segments: [], language: nil)
    }
}

final class ResidentModelTests: XCTestCase {

    private let nul = URL(fileURLWithPath: "/dev/null")

    private func makeModel(
        offload: OffloadPolicy,
        clock: any Clock<Duration> = ContinuousClock()
    ) -> ResidentFullContextModel<CountingMock> {
        ResidentFullContextModel<CountingMock>(
            modelURL: nul, piecesURL: nul, offload: offload, clock: clock)
    }

    /// Poll the actor's live `state` until it reaches `target` (fails on
    /// timeout). Polls the actor rather than the `states` stream so a stale
    /// buffered value (e.g. the initial `.offloaded`) can't match early. The
    /// real-time sleeps only give the idle task a chance to run — the idle
    /// *window* itself is driven by the injected `ManualClock`.
    private func awaitState(
        _ model: any ResidentModel,
        _ target: SpeechModelState,
        timeout: Duration = .seconds(5)
    ) async {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        while clock.now < deadline {
            if await model.state == target { return }
            try? await Task.sleep(for: .milliseconds(5))
        }
        let s = await model.state
        XCTAssertEqual(s, target, "state never reached \(target) within \(timeout)")
    }

    /// Wait until the model's idle timer has registered a sleeper on the manual
    /// clock (so its deadline is fixed at the current time before we advance).
    private func waitForTimer(_ clock: ManualClock) async {
        for _ in 0..<1000 where clock.liveSleepers == 0 {
            await Task.yield()
            try? await Task.sleep(for: .milliseconds(2))
        }
        XCTAssertGreaterThan(clock.liveSleepers, 0, "idle timer never registered")
    }

    // (a) load → transcriber() twice reuses the engine (init runs once).
    func testTranscriberReusesEngine() async throws {
        initCounter.reset()
        let model = makeModel(offload: .pinned)
        await model.warmUp()
        let t1 = try await model.transcriber(locale: nil)
        let t2 = try await model.transcriber(locale: nil)
        XCTAssertEqual(initCounter.count, 1, "engine should load exactly once")
        withExtendedLifetime((t1, t2)) {}
    }

    // (b) idle timer with a fast clock → .offloaded; next transcriber() reloads.
    func testIdleOffloadAndTransparentReload() async throws {
        initCounter.reset()
        let clock = ManualClock()
        let model = makeModel(
            offload: OffloadPolicy(idle: .seconds(100), onMemoryPressure: false), clock: clock)
        await model.warmUp()
        do { let s = await model.state; XCTAssertEqual(s, .ready) }

        await waitForTimer(clock)
        clock.advance(by: .seconds(101))              // trip the idle timer
        await awaitState(model, .offloaded)
        do { let s = await model.state; XCTAssertEqual(s, .offloaded) }
        XCTAssertEqual(initCounter.count, 1)

        let t = try await model.transcriber(locale: nil)   // transparent reload
        XCTAssertEqual(initCounter.count, 2, "next use should reload the engine")
        do { let s = await model.state; XCTAssertEqual(s, .ready) }
        withExtendedLifetime(t) {}
    }

    // (c) activity within the window resets the timer.
    func testActivityResetsIdleTimer() async throws {
        initCounter.reset()
        let clock = ManualClock()
        let model = makeModel(
            offload: OffloadPolicy(idle: .seconds(100), onMemoryPressure: false), clock: clock)
        await model.warmUp()
        await waitForTimer(clock)                      // timer1 registered @ deadline 100

        clock.advance(by: .seconds(60))               // < idle — no offload yet
        // Activity: a full transcriber lifecycle resets the timer on release.
        let t = try await model.transcriber(locale: nil)
        try await t.finalize()
        await waitForTimer(clock)                      // timer2 registered @ deadline 160

        clock.advance(by: .seconds(60))               // now 120 < 160 — reset kept it warm
        do { let s = await model.state; XCTAssertEqual(s, .ready, "reset timer must keep it warm") }
        XCTAssertEqual(initCounter.count, 1, "no reload happened")

        clock.advance(by: .seconds(60))               // now 180 ≥ 160 — past the reset deadline
        await awaitState(model, .offloaded)
    }

    // (d) forced unload() while a transcriber is outstanding is deferred.
    func testUnloadDefersWhileTranscriberOutstanding() async throws {
        initCounter.reset()
        let model = makeModel(offload: .pinned)
        await model.warmUp()
        let t = try await model.transcriber(locale: nil)

        await model.unload()                          // in-use > 0 → deferred
        do { let s = await model.state; XCTAssertEqual(s, .ready, "must not offload under a live transcriber") }

        try await t.finalize()                        // last release → the deferred offload fires
        await awaitState(model, .offloaded)
    }

    // (e) concurrent reloads coalesce into one specialize+load.
    func testConcurrentReloadsCoalesce() async throws {
        initCounter.reset()
        let model = makeModel(offload: .pinned)       // starts offloaded (no warmUp)
        let held = try await withThrowingTaskGroup(of: (any Transcribing).self) { group -> [any Transcribing] in
            for _ in 0..<8 {
                group.addTask { try await model.transcriber(locale: nil) }
            }
            var out: [any Transcribing] = []
            for try await t in group { out.append(t) }
            return out
        }
        XCTAssertEqual(initCounter.count, 1, "8 concurrent loads should coalesce to one")
        withExtendedLifetime(held) {}
    }

    // (f) the vended transcriber releases its pin on finalize/cancel AND via deinit.
    func testPinReleasedOnFinalize() async throws {
        initCounter.reset()
        let model = makeModel(offload: .pinned)
        await model.warmUp()
        let t = try await model.transcriber(locale: nil)
        try await t.finalize()                        // releases the pin
        await model.unload()                          // in-use == 0 → immediate offload
        do { let s = await model.state; XCTAssertEqual(s, .offloaded) }
    }

    func testPinReleasedOnCancel() async throws {
        initCounter.reset()
        let model = makeModel(offload: .pinned)
        await model.warmUp()
        let t = try await model.transcriber(locale: nil)
        await t.cancel()
        await model.unload()
        do { let s = await model.state; XCTAssertEqual(s, .offloaded) }
    }

    func testPinReleasedByDeinitBackstop() async throws {
        initCounter.reset()
        let model = makeModel(offload: .pinned)
        await model.warmUp()
        do {
            let t = try await model.transcriber(locale: nil)
            withExtendedLifetime(t) {}
        }                                             // t dropped WITHOUT finalize/cancel
        // The deinit backstop releases the pin on a detached task; once it lands,
        // the deferred unload can offload. If the backstop were missing, the pin
        // would stay held forever and this would time out.
        await model.unload()
        await awaitState(model, .offloaded)
    }

    // (g) warm-on-load / warmUp() drives .loading → .warming → .ready.
    func testWarmUpTransitions() async throws {
        initCounter.reset()
        let model = makeModel(offload: .pinned)
        let stream = model.states
        let collector = Task { () -> [SpeechModelState] in
            var seen: [SpeechModelState] = []
            for await s in stream {
                seen.append(s)
                if s == .ready { break }
            }
            return seen
        }
        await model.warmUp()
        let seen = await collector.value
        do { let s = await model.state; XCTAssertEqual(s, .ready) }
        XCTAssertTrue(seen.contains(.loading), "should pass through .loading")
        XCTAssertTrue(seen.contains(.warming), "should pass through .warming")
        XCTAssertEqual(seen.last, .ready)
    }

    // MARK: - Cancellation / preemption (SlowCancellableMock)

    private func makeSlowModel() -> ResidentFullContextModel<SlowCancellableMock> {
        // .pinned: residency stays until we explicitly unload(), so an idle timer
        // never interferes with the cancel/supersede assertions.
        ResidentFullContextModel<SlowCancellableMock>(
            modelURL: nul, piecesURL: nul, offload: .pinned, clock: ContinuousClock())
    }

    /// A 16 kHz mono Float32 buffer of `frames` samples (content irrelevant —
    /// the mock ignores audio). Lands on the converter's fast path.
    private func makeInput(frames: Int) -> AnalyzerInput {
        let fmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
            channels: 1, interleaved: false)!
        let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(frames))!
        buf.frameLength = AVAudioFrameCount(frames)
        buf.floatChannelData![0].update(repeating: 0.01, count: frames)
        return AnalyzerInput(buffer: buf, bufferStartTime: .zero)
    }

    /// Collect every result a transcriber emits until its stream finishes.
    private func collectResults(_ t: any Transcribing) -> Task<[TranscriptionResult], Never> {
        let stream = t.results
        return Task {
            var out: [TranscriptionResult] = []
            do { for try await r in stream { out.append(r) } } catch { }
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

    // Cancel an in-flight (mock) pass mid-decode: the pass BAILS promptly (not
    // run to completion), the results stream finishes with NO final emitted, and
    // the pin is released (unload → offloaded).
    func testCancelInFlightPassBailsPromptlyAndReleasesPin() async throws {
        initCounter.reset(); slowControl.reset()
        let model = makeSlowModel()
        let t = try await model.transcriber(locale: nil)
        try await t.prepare()
        let collector = collectResults(t)

        // ≥ refreshStride (700 ms = 11 200) but < commitThreshold → the plain
        // refresh branch, which dispatches one engine.transcribe pass.
        try await t.consume(makeInput(frames: 12_000))
        await poll { slowControl.inFlight > 0 }
        XCTAssertEqual(slowControl.inFlight, 1, "a refresh pass should be in flight")

        await t.cancel()                          // prompt cancel
        await poll { slowControl.bailed == 1 }
        XCTAssertEqual(slowControl.bailed, 1, "the in-flight pass must bail on cancel")
        XCTAssertEqual(slowControl.completed, 0, "no pass should run to completion")

        let results = await collector.value       // finishes because cancel closed the stream
        XCTAssertFalse(results.contains { $0.isFinal }, "no final result for a cancelled job")

        await model.unload()                      // pin released → offload fires
        await awaitState(model, .offloaded)
    }

    // Supersede: cancel a live job, then vend a NEW transcriber on the SAME
    // resident model. The new job produces a COMPLETE final result, the engine
    // was loaded exactly once (no reload), and no pin leaks (unload → offloaded).
    func testSupersedeReusesWarmEngineAndCompletes() async throws {
        initCounter.reset(); slowControl.reset()
        let model = makeSlowModel()

        let t1 = try await model.transcriber(locale: nil)
        try await t1.prepare()
        let c1 = collectResults(t1)
        try await t1.consume(makeInput(frames: 12_000))
        await poll { slowControl.inFlight > 0 }
        await t1.cancel()

        // New job on the same warm engine.
        slowControl.open()                        // let its finalize pass complete
        let t2 = try await model.transcriber(locale: nil)
        try await t2.prepare()
        let c2 = collectResults(t2)
        try await t2.consume(makeInput(frames: 4_000))   // < refreshStride: no partial pass
        try await t2.finalize()

        let r1 = await c1.value
        XCTAssertFalse(r1.contains { $0.isFinal }, "cancelled job emits no final")
        let r2 = await c2.value
        let final = r2.last { $0.isFinal }
        XCTAssertNotNil(final, "superseding job should emit a final")
        XCTAssertEqual(String(final!.text.characters), "complete transcript",
                       "the new job's result should be complete")

        XCTAssertEqual(initCounter.count, 1, "engine must be reused — no reload")
        await model.unload()                      // no leaked pin
        await awaitState(model, .offloaded)
    }

    // Single-active policy: requesting a second transcriber while one is live
    // AUTO-CANCELS the first (supersede). The first job's in-flight pass bails
    // and it emits no final; the second is usable.
    func testSecondTranscriberSupersedesFirst() async throws {
        initCounter.reset(); slowControl.reset()
        let model = makeSlowModel()

        let t1 = try await model.transcriber(locale: nil)
        try await t1.prepare()
        let c1 = collectResults(t1)
        try await t1.consume(makeInput(frames: 12_000))
        await poll { slowControl.inFlight > 0 }

        // Do NOT cancel t1 explicitly — requesting t2 should supersede it.
        let t2 = try await model.transcriber(locale: nil)
        await poll { slowControl.bailed == 1 }
        XCTAssertEqual(slowControl.bailed, 1, "requesting a new transcriber cancels the prior")

        let r1 = await c1.value                   // t1's stream was closed by the supersede
        XCTAssertFalse(r1.contains { $0.isFinal }, "superseded job emits no final")
        XCTAssertEqual(initCounter.count, 1, "same warm engine — no reload")
        await t2.cancel()                         // release t2's pin so unload can fire
        await model.unload()
        await awaitState(model, .offloaded)
    }
}
