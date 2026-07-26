// VoixfulSpeech — the resident (warm, reusable, auto-offloading) model surface.
//
// A `ResidentModel` is "a loaded model" separated from "one utterance's
// transcriber": load + graph-specialize once, vend a fresh `Transcribing` per
// utterance into the SAME `VoixfulAnalyzer` flow, and auto-offload the weights
// after idle / under memory pressure (reloading transparently on next use).
//
// This file is the model-agnostic core: the protocol, the state/policy value
// types, the leak-safe transcriber wrapper, and the generic lifecycle actor
// (`ResidentModelActor`) shared by every backend. It deliberately does NOT
// reference `ModelBackend`/`VoixfulEngine`, so the pure brand core stays free of
// the engine layer — the backend-specific loading lives behind the tiny
// `ResidentModelBackend` seam, implemented in each backend's own module.

import Dispatch
import Foundation

// MARK: - State & policy

/// Observable lifecycle of a `ResidentModel`, surfaced on `states` so a host can
/// render "Warming… / Idle / Ready" instead of a frozen UI on a transparent
/// reload.
public enum SpeechModelState: Sendable, Equatable {
    /// Weights freed; the next use reloads transparently.
    case offloaded
    /// Asset open + weight materialization in flight.
    case loading
    /// First-pass Core AI graph specialization in flight (the multi-second cost).
    case warming
    /// Warm; per-utterance cost is minimal.
    case ready
}

/// When a resident model reclaims its RAM on its own.
public struct OffloadPolicy: Sendable {
    /// Idle window before auto-offload; `nil` never idle-offloads.
    public var idle: Duration?
    /// Offload early when the OS raises memory pressure (gated on no live
    /// transcriber — see `ResidentModelActor`).
    public var onMemoryPressure: Bool

    public init(idle: Duration?, onMemoryPressure: Bool) {
        self.idle = idle
        self.onMemoryPressure = onMemoryPressure
    }

    /// Idle 120 s + memory-pressure offload — the sensible default for a
    /// dictation app that reuses one model across bursts of utterances.
    public static let `default` = OffloadPolicy(idle: .seconds(120), onMemoryPressure: true)
    /// Never offloads — the model stays resident until explicitly `unload()`ed.
    public static let pinned = OffloadPolicy(idle: nil, onMemoryPressure: false)
}

// MARK: - The public façade

/// A loaded, warm, reusable model that vends a ready transcriber into the
/// existing analyzer flow. Model-agnostic: `transcriber(locale:)` returns the
/// same `any Transcribing` callers already drive, so adopting residency swaps
/// *where the transcriber comes from*, nothing about how you transcribe.
///
/// The concrete conformers (`ResidentModelActor`) are `package`; hosts obtain
/// one via `ResidentModel.load(...)` (in the umbrella `VoixfulKit` module) and
/// hold it as `any ResidentModel`.
public protocol ResidentModel: Actor {
    /// Current lifecycle state.
    var state: SpeechModelState { get }
    /// Stream of state transitions for an optional HUD.
    nonisolated var states: AsyncStream<SpeechModelState> { get }

    /// Pay first-pass graph specialization now (also done on `load`).
    func warmUp() async
    /// Free the model's RAM now; idempotent. Defers until any outstanding
    /// transcriber releases (freeing mid-utterance reclaims nothing via ARC).
    func unload() async

    /// A fresh transcriber wrapping the SHARED warm engine — drop it into a
    /// `VoixfulAnalyzer` exactly like today. Reloads transparently if the model
    /// was auto-offloaded. Auto-releases its RAM "pin" on `finalize()`/`cancel()`,
    /// with a `deinit` backstop so a forgotten teardown can't strand the model.
    func transcriber(locale: Locale?) async throws -> any Transcribing
}

extension ResidentModel {
    /// Vend a transcriber in the model's configured locale.
    public func transcriber() async throws -> any Transcribing {
        try await transcriber(locale: nil)
    }
}

// MARK: - Backend seam

/// The three backend-specific operations the generic lifecycle needs. Each
/// backend module provides one concrete conformer (full-context, Nemotron), so
/// the residency/offload/concurrency machinery lives in ONE audited place.
public protocol ResidentModelBackend: Sendable {
    /// The loaded model handle (an engine actor — a reference type retained by
    /// live transcribers via ARC).
    associatedtype Engine: Sendable
    /// The concrete transcriber this backend vends (wrapped leak-safe before it
    /// leaves the actor).
    associatedtype Vended: Transcribing

    /// Open the asset + materialize weights (`prewarm`). The expensive first-pass
    /// specialization is paid separately in `warm`.
    func load() async throws -> Engine
    /// Run first-pass specialization off the hot path (silent buffer / SOS reset).
    func warm(_ engine: Engine) async
    /// Build a fresh, single-use transcriber around the shared engine.
    func makeTranscriber(_ engine: Engine, locale: Locale?) -> Vended
}

// MARK: - Leak-safe transcriber wrapper

/// A thin wrapper around a backend's real transcriber that releases the resident
/// model's in-use "pin" on the FIRST of `finalize()`/`cancel()` — with a
/// `deinit` backstop so a caller that forgets to tear down can't pin the model's
/// RAM forever. This is the single correctness detail that makes the simple
/// `transcriber()` API leak-safe.
package final class PinnedTranscriber<Inner: Transcribing>: Transcribing, @unchecked Sendable {

    public static var supportedLocales: [Locale] { Inner.supportedLocales }

    private let inner: Inner
    private let onRelease: @Sendable () async -> Void
    private let lock = NSLock()
    private var releaseClaimed = false

    // Transcribing surface forwards straight to the inner transcriber. These are
    // all synchronous protocol requirements (the conformers expose them as
    // `nonisolated let`), so reading them through `Inner` needs no await.
    public var moduleIdentifier: String { inner.moduleIdentifier }
    public var modelName: String { inner.modelName }
    public var locale: Locale { inner.locale }
    public var reportingOptions: ReportingOption { inner.reportingOptions }
    public var attributeOptions: ResultAttributeOption { inner.attributeOptions }
    public var results: AsyncThrowingStream<TranscriptionResult, Error> { inner.results }

    package init(wrapping inner: Inner, onRelease: @escaping @Sendable () async -> Void) {
        self.inner = inner
        self.onRelease = onRelease
    }

    public func prepare() async throws { try await inner.prepare() }
    public func consume(_ input: AnalyzerInput) async throws { try await inner.consume(input) }

    public func finalize() async throws {
        do {
            try await inner.finalize()
        } catch {
            await releaseOnce()
            throw error
        }
        await releaseOnce()
    }

    public func cancel() async {
        await inner.cancel()
        await releaseOnce()
    }

    deinit {
        // Backstop: the caller dropped us without finalize()/cancel(). Free the
        // pin on a detached task (deinit can't await) so the model isn't stranded.
        if claimRelease() {
            let onRelease = self.onRelease
            Task.detached { await onRelease() }
        }
    }

    private func releaseOnce() async {
        if claimRelease() { await onRelease() }
    }

    /// True exactly once — guards against a double-decrement across
    /// finalize/cancel and the deinit backstop.
    private func claimRelease() -> Bool {
        lock.lock(); defer { lock.unlock() }
        if releaseClaimed { return false }
        releaseClaimed = true
        return true
    }
}

// MARK: - Generic lifecycle actor

/// The shared residency/offload/reload machinery, generic over a
/// `ResidentModelBackend`. Each backend module exposes it under a friendly name
/// (`ResidentFullContextModel<Engine>`, `ResidentNemotronModel`) via a typealias
/// + a convenience initializer.
///
/// Concurrency invariants (the correctness core):
///   1. `transcriber()` increments the in-use count SYNCHRONOUSLY before its
///      first `await`, so the idle timer can't offload mid-load.
///   2. `unload()` / idle / memory-pressure offload DEFER to in-use == 0 — a
///      live transcriber retains the engine via ARC, so freeing mid-utterance
///      reclaims nothing (and risks two ~3.7 GB generations).
///   3. The in-flight load is memoized (`loadTask`) so concurrent calls coalesce
///      into ONE specialize+prewarm.
///   4. The idle-timer task holds `self` weakly, so a dropped model still deinits.
package actor ResidentModelActor<Backend: ResidentModelBackend>: ResidentModel {

    private let backend: Backend
    private let offload: OffloadPolicy
    private let clock: any Clock<Duration>

    private var engine: Backend.Engine?
    private var loadTask: Task<Backend.Engine, Error>?
    private var inUse = 0
    private var pendingOffload = false
    private var idleTimer: Task<Void, Never>?

    private var _state: SpeechModelState = .offloaded
    private nonisolated let stateContinuation: AsyncStream<SpeechModelState>.Continuation
    public nonisolated let states: AsyncStream<SpeechModelState>

    // Set once in init, cancelled once in deinit; never mutated concurrently.
    private nonisolated let memoryPressure: DispatchSourceMemoryPressure?

    public var state: SpeechModelState { _state }

    package init(
        backend: Backend,
        offload: OffloadPolicy = .default,
        clock: any Clock<Duration> = ContinuousClock()
    ) {
        self.backend = backend
        self.offload = offload
        self.clock = clock
        let (stream, continuation) = AsyncStream<SpeechModelState>.makeStream()
        self.states = stream
        self.stateContinuation = continuation
        continuation.yield(.offloaded)

        if offload.onMemoryPressure {
            let source = DispatchSource.makeMemoryPressureSource(
                eventMask: [.warning, .critical],
                queue: DispatchQueue.global(qos: .utility))
            self.memoryPressure = source
            source.setEventHandler { [weak self] in
                guard let self else { return }
                Task { await self.offloadOrDefer() }
            }
            source.activate()
        } else {
            self.memoryPressure = nil
        }
    }

    deinit {
        memoryPressure?.cancel()
        stateContinuation.finish()
    }

    // MARK: Public API

    public func warmUp() async {
        inUse += 1                              // pin before the first await (#1)
        cancelIdleTimer()
        defer { release() }
        do {
            let engine = try await loadedEngine()
            setState(.warming)
            await backend.warm(engine)
            setState(.ready)
        } catch {
            // load() failed; loadedEngine() already reset state to .offloaded.
        }
    }

    public func unload() async {
        offloadOrDefer()
    }

    public func transcriber(locale: Locale?) async throws -> any Transcribing {
        inUse += 1                              // pin before the first await (#1)
        cancelIdleTimer()
        do {
            let engine = try await loadedEngine()
            if _state != .ready { setState(.ready) }
            let inner = backend.makeTranscriber(engine, locale: locale)
            return PinnedTranscriber(wrapping: inner) { [weak self] in
                await self?.release()
            }
        } catch {
            release()                            // undo the pin on a failed load
            throw error
        }
    }

    // MARK: Load (single-flighted — #3)

    private func loadedEngine() async throws -> Backend.Engine {
        if let engine { return engine }
        if let loadTask { return try await loadTask.value }

        setState(.loading)
        let backend = self.backend
        let task = Task<Backend.Engine, Error> { try await backend.load() }
        loadTask = task
        do {
            let created = try await task.value
            engine = created
            loadTask = nil
            return created
        } catch {
            loadTask = nil
            setState(.offloaded)
            throw error
        }
    }

    // MARK: Offload / release

    /// Decrement the in-use count; on reaching zero, either honor a pending
    /// offload or (re)arm the idle timer.
    private func release() {
        inUse = max(0, inUse - 1)
        guard inUse == 0 else { return }
        if pendingOffload {
            pendingOffload = false
            performOffload()
        } else {
            startIdleTimer()
        }
    }

    /// Offload now if idle, else mark it pending so the last release frees the
    /// weights (#2). Shared by `unload()`, memory pressure, and the idle timer.
    private func offloadOrDefer() {
        guard inUse == 0 else { pendingOffload = true; return }
        performOffload()
    }

    private func performOffload() {
        engine = nil                            // ARC frees the weights (no live transcriber holds them)
        loadTask = nil
        cancelIdleTimer()
        setState(.offloaded)
    }

    // MARK: Idle timer (#4 — weak self)

    private func startIdleTimer() {
        cancelIdleTimer()
        guard let idle = offload.idle, engine != nil, inUse == 0 else { return }
        idleTimer = Task { [weak self, clock, idle] in
            do {
                try await Task.sleep(for: idle, tolerance: nil, clock: clock)
            } catch {
                return                           // cancelled (reset by activity)
            }
            if Task.isCancelled { return }
            await self?.idleFired()
        }
    }

    private func idleFired() {
        guard inUse == 0, engine != nil else { return }
        performOffload()
    }

    private func cancelIdleTimer() {
        idleTimer?.cancel()
        idleTimer = nil
    }

    // MARK: State

    private func setState(_ next: SpeechModelState) {
        guard next != _state else { return }
        _state = next
        stateContinuation.yield(next)
    }
}
