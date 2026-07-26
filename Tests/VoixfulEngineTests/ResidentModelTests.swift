// Deterministic residency tests — NO real model. A mock FullContextEngine with a
// shared init counter (like LongFormMockTests) verifies the ResidentModel
// lifecycle: engine reuse, idle offload + transparent reload, activity reset,
// deferred offload while a transcriber is outstanding, coalesced concurrent
// reloads, pin release on finalize/cancel AND via deinit, and warm-on-load.
//
// The idle timer runs on an injected `ManualClock` so time advances only when
// the test says so — no wall-clock flakiness.

import XCTest
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
        _ model: ResidentFullContextModel<CountingMock>,
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
}
