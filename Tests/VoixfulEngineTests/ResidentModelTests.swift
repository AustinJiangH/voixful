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
        // Block indefinitely until the gate opens (complete) or the Task is
        // cancelled (bail) — no wall-clock cap. XCTest's own timeout is the hang
        // guard; a bounded spin uncoordinated with the test's real-time polling
        // would be a latent CI flake.
        while true {
            if Task.isCancelled { slowControl.exitBailed(); throw CancellationError() }
            if slowControl.isOpen {
                slowControl.exitCompleted()
                return Transcript(text: "complete transcript", segments: [], language: nil)
            }
            try? await Task.sleep(for: .milliseconds(2))
        }
    }
}

// MARK: - Gated mock engine (blocks load until the test releases it)

/// Gate for `GatedMock`: `load()` blocks until `release()` so N concurrent
/// `transcriber()` calls all pile onto the single memoized load — a non-memoized
/// impl would instead run `init` N times.
private final class LoadGate: @unchecked Sendable {
    private let lock = NSLock()
    private var _released = false
    var isReleased: Bool { lock.lock(); defer { lock.unlock() }; return _released }
    func release() { lock.lock(); _released = true; lock.unlock() }
    func reset() { lock.lock(); _released = false; lock.unlock() }
}
private let loadGate = LoadGate()

private actor GatedMock: FullContextEngine {
    static let displayName = "GatedMock"
    static let identifierSuffix = "gatedmock"
    static var supportedLocales: [Locale] { [Locale(identifier: "en-US")] }
    nonisolated let maxAudioSamples = 16_000

    init(modelURL: URL, piecesURL: URL) async throws {
        initCounter.bump()
        while !loadGate.isReleased { try? await Task.sleep(for: .milliseconds(2)) }
    }
    func prewarm() {}
    func transcribe(audio: [Float]) async throws -> Transcript {
        Transcript(text: "", segments: [], language: nil)
    }
}

// MARK: - Throwing mock engine (fails the first N loads, then succeeds)

/// Controls how many `GatedMock`-free `ThrowingMock` loads throw before one
/// succeeds — to exercise the load-failure retry / no-pin-leak path.
private final class ThrowControl: @unchecked Sendable {
    private let lock = NSLock()
    private var _attempts = 0
    private var _failFirst = 0
    func setFailures(_ n: Int) { lock.lock(); _failFirst = n; _attempts = 0; lock.unlock() }
    /// Called once per load attempt; true while the attempt should throw.
    func shouldThrow() -> Bool {
        lock.lock(); defer { lock.unlock() }
        _attempts += 1
        return _attempts <= _failFirst
    }
}
private let throwControl = ThrowControl()

private actor ThrowingMock: FullContextEngine {
    static let displayName = "ThrowMock"
    static let identifierSuffix = "throwmock"
    static var supportedLocales: [Locale] { [Locale(identifier: "en-US")] }
    nonisolated let maxAudioSamples = 16_000

    init(modelURL: URL, piecesURL: URL) async throws {
        initCounter.bump()
        if throwControl.shouldThrow() { throw VoixfulError.engine("simulated load failure") }
    }
    func prewarm() {}
    func transcribe(audio: [Float]) async throws -> Transcript {
        Transcript(text: "", segments: [], language: nil)
    }
}

// MARK: - Thread-safe recorder for the states stream

private final class StatesRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var _states: [SpeechModelState] = []
    func append(_ s: SpeechModelState) { lock.lock(); _states.append(s); lock.unlock() }
    var snapshot: [SpeechModelState] { lock.lock(); defer { lock.unlock() }; return _states }
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

    // (d) forced unload() while a transcriber is outstanding is deferred — and
    // the engine stays genuinely usable under the deferral (not just the label):
    // we drive the transcriber to a real final result before the deferred offload
    // fires on its last release.
    func testUnloadDefersWhileTranscriberOutstanding() async throws {
        initCounter.reset()
        let model = makeModel(offload: .pinned)
        await model.warmUp()
        let t = try await model.transcriber(locale: nil)

        await model.unload()                          // in-use > 0 → deferred
        do { let s = await model.state; XCTAssertEqual(s, .ready, "must not offload under a live transcriber") }

        // Prove the shared engine still works while the unload is deferred.
        try await t.prepare()
        try await t.consume(makeInput(frames: 4_000))
        let collector = collectResults(t)
        try await t.finalize()                        // last release → the deferred offload fires
        let results = await collector.value
        XCTAssertTrue(results.contains { $0.isFinal }, "engine must stay usable under a deferred unload")
        await awaitState(model, .offloaded)
    }

    // (e) concurrent reloads coalesce into one specialize+load. Deterministic:
    // GatedMock.load() blocks until every vend has pinned (all N piled onto the
    // one memoized loadTask), THEN the gate opens — so a non-memoized impl would
    // yield init == N, and coalescing shows init == 1. We do NOT hold N live
    // transcribers (that would violate single-active); the vends are discarded.
    func testConcurrentReloadsCoalesce() async throws {
        initCounter.reset(); loadGate.reset()
        let model = ResidentFullContextModel<GatedMock>(
            modelURL: nul, piecesURL: nul, offload: .pinned, clock: ContinuousClock())
        let n = 8
        let group = Task {
            try await withThrowingTaskGroup(of: Void.self) { g in
                for _ in 0..<n { g.addTask { _ = try await model.transcriber(locale: nil) } }
                try await g.waitForAll()
            }
        }
        // Wait until all n vends have pinned — i.e. all are awaiting the single
        // in-flight (gated) load. Only then release, so the coalescing path is
        // the one exercised.
        await pollAsync { await model.inUseCount == n }
        do { let c = await model.inUseCount; XCTAssertEqual(c, n, "all \(n) vends should pin before the load resolves") }
        loadGate.release()
        try await group.value
        XCTAssertEqual(initCounter.count, 1, "\(n) concurrent loads should coalesce to one")
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
        let iLoading = seen.firstIndex(of: .loading)
        let iWarming = seen.firstIndex(of: .warming)
        XCTAssertNotNil(iLoading, "should pass through .loading")
        XCTAssertNotNil(iWarming, "should pass through .warming")
        XCTAssertLessThan(iLoading!, iWarming!, ".loading must precede .warming")
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

    /// `poll` for an async predicate (e.g. reading an actor's state).
    private func pollAsync(_ timeout: Duration = .seconds(5), until: () async -> Bool) async {
        let clock = ContinuousClock()
        let deadline = clock.now.advanced(by: timeout)
        while clock.now < deadline {
            if await until() { return }
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

    // Single-active under CONCURRENT vends (proves the fix for finding #1): two
    // transcriber() calls that interleave across the load await must NOT both go
    // live on the shared engine. After both vends return, the loser is already
    // cancelled (the last installer awaited the prior's cancel), so driving BOTH
    // yields at most ONE engine pass in flight. No pin leaks (unload → offloaded).
    func testConcurrentVendsAreSingleActive() async throws {
        initCounter.reset(); slowControl.reset()
        let model = makeSlowModel()                // starts offloaded → both race the load

        async let aVend = model.transcriber(locale: nil)
        async let bVend = model.transcriber(locale: nil)
        let ta = try await aVend
        let tb = try await bVend

        try await ta.prepare(); try await tb.prepare()
        let ca = collectResults(ta); let cb = collectResults(tb)
        // Drive both. The cancelled (superseded) one gates maybeRefresh on its
        // `cancelled` flag, so it starts no pass — only the live one dispatches.
        try await ta.consume(makeInput(frames: 12_000))
        try await tb.consume(makeInput(frames: 12_000))
        await poll { slowControl.inFlight >= 1 }
        XCTAssertEqual(slowControl.inFlight, 1,
                       "at most one transcriber may ever drive the shared engine")
        XCTAssertEqual(initCounter.count, 1, "coalesced load — one generation only")

        await ta.cancel(); await tb.cancel()      // live pass bails; both pins released
        _ = await ca.value; _ = await cb.value
        await model.unload()
        await awaitState(model, .offloaded)
    }

    // Load THROWS → retry + no pin leak (finding #7). First load throws; the vend
    // rethrows and leaves the model .offloaded with its pin released (loadTask
    // cleared). A subsequent vend succeeds (proving retry), and a final unload
    // offloads (proving the failed load leaked no pin — no deadlock).
    func testLoadThrowsThenRetriesWithoutPinLeak() async throws {
        initCounter.reset(); throwControl.setFailures(1)
        let model = ResidentFullContextModel<ThrowingMock>(
            modelURL: nul, piecesURL: nul, offload: .pinned, clock: ContinuousClock())

        do {
            _ = try await model.transcriber(locale: nil)
            XCTFail("first vend should throw when the load fails")
        } catch { /* expected */ }
        do { let s = await model.state; XCTAssertEqual(s, .offloaded, "failed load resets to .offloaded") }
        do { let n = await model.inUseCount; XCTAssertEqual(n, 0, "failed load must release its pin") }

        // Retry: loadTask was cleared, so this reloads and succeeds.
        let t = try await model.transcriber(locale: nil)
        do { let s = await model.state; XCTAssertEqual(s, .ready) }
        XCTAssertEqual(initCounter.count, 2, "one failed load + one successful load")
        try await t.finalize()

        await model.unload()                      // no leaked pin / deadlock
        await awaitState(model, .offloaded)
    }

    // Memory-pressure offload (finding #7): pressure with a live transcriber
    // DEFERS (ARC would reclaim nothing); pressure with none offloads immediately.
    // simulateMemoryPressure() runs the same offloadOrDefer path as the OS handler.
    func testMemoryPressureDefersUnderLiveTranscriberElseOffloads() async throws {
        initCounter.reset()
        let model = makeModel(offload: .pinned)   // .pinned: no idle timer to interfere

        // (1) Live transcriber → pressure defers.
        await model.warmUp()
        let t = try await model.transcriber(locale: nil)
        await model.simulateMemoryPressure()
        do { let s = await model.state; XCTAssertEqual(s, .ready, "pressure must defer under a live transcriber") }
        try await t.finalize()                    // last release → the deferred offload fires
        await awaitState(model, .offloaded)

        // (2) No live transcriber → pressure offloads immediately.
        await model.warmUp()
        do { let s = await model.state; XCTAssertEqual(s, .ready) }
        await model.simulateMemoryPressure()
        await awaitState(model, .offloaded)
    }

    // Transparent reload skips re-specialization (finding #7): across an idle
    // offload → reload via transcriber(), the states stream emits .loading →
    // .ready and NOT .warming (reload repays only weight materialization, not the
    // multi-second graph specialization — persistent Core AI cache). Locks the
    // contract.
    func testTransparentReloadSkipsWarming() async throws {
        initCounter.reset()
        let clock = ManualClock()
        let model = makeModel(
            offload: OffloadPolicy(idle: .seconds(100), onMemoryPressure: false), clock: clock)
        let recorder = StatesRecorder()
        let stream = model.states
        let collector = Task { for await s in stream { recorder.append(s) } }
        defer { collector.cancel() }

        await model.warmUp()                      // .loading → .warming → .ready
        await awaitState(model, .ready)
        await waitForTimer(clock)
        clock.advance(by: .seconds(101))          // idle → .offloaded
        await awaitState(model, .offloaded)

        let t = try await model.transcriber(locale: nil)   // transparent reload
        await awaitState(model, .ready)
        await poll { recorder.snapshot.last == .ready }

        let seen = recorder.snapshot
        guard let lastOffload = seen.lastIndex(of: .offloaded) else {
            return XCTFail("model should have idle-offloaded before the reload")
        }
        let reloadTail = Array(seen[(lastOffload + 1)...])
        XCTAssertEqual(reloadTail, [.loading, .ready], "reload is .loading → .ready")
        XCTAssertFalse(reloadTail.contains(.warming), "transparent reload must skip re-specialization")
        withExtendedLifetime(t) {}
    }

    // Double-cancel / cancel-after-finalize idempotence (finding #7): the terminal
    // release fires exactly once across finalize() then cancel() (and cancel()
    // twice); inUse never underflows, and the lifecycle stays intact afterward.
    func testTerminalReleaseIsIdempotent() async throws {
        initCounter.reset()
        let model = makeModel(offload: .pinned)
        await model.warmUp()
        let t = try await model.transcriber(locale: nil)

        try await t.finalize()                    // releases the pin (once)
        await t.cancel()                          // once-guard: no second release
        await t.cancel()                          // still no-op
        do { let n = await model.inUseCount; XCTAssertEqual(n, 0, "inUse must not underflow past zero") }

        await model.unload()
        await awaitState(model, .offloaded)

        // Engine lifecycle is intact: a fresh vend still works.
        let t2 = try await model.transcriber(locale: nil)
        do { let s = await model.state; XCTAssertEqual(s, .ready) }
        await t2.cancel()
        await model.unload()
        await awaitState(model, .offloaded)
    }

    // .pinned never idle-offloads (finding #7): under .pinned (idle == nil) no idle
    // timer is ever armed, so advancing the clock far past any window keeps .ready.
    func testPinnedNeverIdleOffloads() async throws {
        initCounter.reset()
        let clock = ManualClock()
        let model = makeModel(offload: .pinned, clock: clock)   // idle == nil
        await model.warmUp()
        do { let s = await model.state; XCTAssertEqual(s, .ready) }
        XCTAssertEqual(clock.liveSleepers, 0, ".pinned must not arm an idle timer")

        clock.advance(by: .seconds(100_000))      // well past any plausible idle window
        try? await Task.sleep(for: .milliseconds(20))   // give any errant idle task a chance
        do { let s = await model.state; XCTAssertEqual(s, .ready, ".pinned must never idle-offload") }
        XCTAssertEqual(initCounter.count, 1, "no reload — the model stayed resident")
    }

    // Nemotron residency + per-utterance streaming reset.
    //
    // The contract (design §Cancellation.4): a vended Nemotron transcriber's
    // prepare() must run engine.start() (SOS reset) so utterance #2 can't inherit
    // #1's shared streaming cache. That branch lives in
    // `NemotronStreamingTranscriber.prepare()` (engineInjected && mode==.streaming
    // → await engine.start()) and is exercised end-to-end by the Nemotron mic path.
    //
    // A DETERMINISTIC unit test of it is not feasible here: NemotronStreamingEngine
    // is a concrete `public actor` whose init loads a real .aimodel (four Core AI
    // sessions), and NemotronStreamingTranscriber holds it as a concrete type with
    // no protocol seam — so no mock engine can be injected without an engine-layer
    // refactor the design explicitly avoids. The generic single-active/supersede
    // machinery that drives the reset (one transcriber per vend, prior cancelled)
    // is covered by the FullContext supersede tests above; the reset itself is a
    // three-line branch verified by inspection. Skipped, not silently omitted.
    func testNemotronPerUtteranceStreamingReset() throws {
        throw XCTSkip("""
            Requires a real Nemotron .aimodel or an engine protocol seam; \
            NemotronStreamingEngine has no injectable mock. Reset branch is \
            covered by inspection + the Nemotron mic path (see NemotronStreamingTranscriber.prepare()).
            """)
    }
}
