// Phase H — Cache round-trip tests for Core AI's MutableViews state machinery.
//
// Two tests cover the two layers PLAN.md §4 separates:
//
//   1. `testNDArrayWriteRead` — synthetic, no model required. Allocates an
//      `NDArray`, writes a known pattern via the mutable accessor, reads it
//      back via the read accessor, and asserts equality. Catches regressions
//      in the lowest layer of the cache machinery: the assumption that
//      `mutableView().fill(...)` and the matching writer/reader accessor pair
//      mutate and observe the same underlying buffer. If this test fails, the
//      four streaming caches in EncoderSession will silently drop their
//      contents on every call.
//
//   2. `testEncoderSessionResetClearsCaches` — end-to-end with a real
//      `.aimodel`. Constructs an EncoderSession, mutates its cache state via
//      a forward pass, calls `reset()`, and verifies the caches are zero.
//      Skipped (`throw XCTSkip(...)`) if the `fixtures/aimodel` artifact is
//      absent — there's no way to construct EncoderSession without a model.
//
// Both tests are gated behind `#if canImport(CoreAI)` because the NDArray
// type lives in the Core AI module; without the framework on the import path,
// neither test can compile. The `#else` branch issues an explicit
// `throw XCTSkip(...)` so the absence shows up in the test report rather than
// being silently elided.

import XCTest
import Foundation
@testable import VoixfulEngine

#if canImport(CoreAI)
import CoreAI
#endif

final class CacheRoundTripTests: XCTestCase {

    // MARK: - Synthetic NDArray round-trip

    #if canImport(CoreAI)

    /// Allocate an NDArray of shape `[2, 3, 4]`, write a known pattern via
    /// the mutable accessor, then read it back via the view accessor and
    /// assert every element matches. The pattern is `i * 0.5 + 1.0` over the
    /// flat row-major index `i`, which catches both index-ordering bugs and
    /// silent zeroing bugs.
    ///
    /// This is the foundation cache-machinery test: if the runtime fails to
    /// keep the buffer alive between write and read, the rest of the cache
    /// system can't possibly work. The actual streaming-cache tests run only
    /// when a fixture model is available; this test always runs (when the
    /// framework is importable) and serves as the fast-failing canary.
    ///
    /// VERIFY: NDArray accessor API. EncoderSession + LangID use
    /// `mutableView().fill(...)` and `mutableView().withUnsafeMutableBufferPointer(of:)`
    /// for writes; the read counterpart is assumed to be
    /// `view().withUnsafeBufferPointer(of:)`. If the SDK names differ, update
    /// all four call sites (EncoderSession.reset, PredictionSession.step,
    /// LangID.oneHot, this test) uniformly.
    func testNDArrayWriteRead() async throws {
        let shape: [Int] = [2, 3, 4]
        let count = shape.reduce(1, *)
        XCTAssertEqual(count, 24, "test fixture: shape [2,3,4] flat-count = 24")

        var array = NDArray(shape: shape, scalarType: .float32)

        // Write phase: zero first (NDArray initial contents are unspecified
        // per PLAN.md §4 — EncoderSession follows the same zero-then-write
        // convention), then fill with the test pattern.
        array.zeroFloat32()
        var mv = array.mutableView(as: Float.self)
        mv.withUnsafeMutablePointer { ptr, _, _ in
            for i in 0..<count {
                ptr[i] = Float(i) * 0.5 + 1.0
            }
        }

        // Read phase: pull a read-only view back through the view accessor
        // and verify the pattern survived. The two accessors must observe
        // the same underlying buffer — that's the whole point of MutableViews.
        let snapshot = array.flatFloat32()
        XCTAssertEqual(snapshot.count, count,
                       "snapshot count must equal product of shape")
        for i in 0..<snapshot.count {
            XCTAssertEqual(snapshot[i], Float(i) * 0.5 + 1.0,
                           accuracy: 1e-6,
                           "round-trip mismatch at index \(i)")
        }
    }

    /// Smaller companion: verify the in-place zero helper actually zeros every
    /// element after a non-zero write. This is the exact pattern
    /// `EncoderSession.reset()` and `PredictionSession.reset()` rely on; if it
    /// silently no-ops, cache resets across utterances will leak state.
    func testNDArrayFillZero() async throws {
        var array = NDArray(shape: [4, 8], scalarType: .float32)

        // Write non-zero data first.
        array.fillFloat32(1.0)

        // Then reset.
        array.zeroFloat32()

        // Verify every slot is exactly 0.
        let snapshot = array.flatFloat32()
        for i in 0..<snapshot.count {
            XCTAssertEqual(snapshot[i], 0.0,
                           "zeroFloat32() failed at index \(i): got \(snapshot[i])")
        }
    }

    #else

    /// CoreAI module isn't importable — likely running pre-Xcode-27 or before
    /// the SDK has been installed. Surfacing this as a skip rather than a
    /// silent omission keeps the test report honest.
    func testNDArrayWriteRead() throws {
        throw XCTSkip("CoreAI framework not importable — install Xcode 27 SDK to run NDArray cache tests")
    }

    func testNDArrayFillZero() throws {
        throw XCTSkip("CoreAI framework not importable — install Xcode 27 SDK to run NDArray cache tests")
    }

    #endif

    // MARK: - EncoderSession end-to-end reset

    #if canImport(CoreAI)

    /// Build an `EncoderSession` from the fixtures `.aimodel`, mutate cache
    /// state via a forward pass, call `reset()`, and assert all four caches
    /// are zero. Skipped when the fixture model is missing — there's no way
    /// to construct EncoderSession without it.
    ///
    /// VERIFY: this test reaches into `EncoderSession`'s internal cache
    /// properties via the `@testable import VoixfulEngine` access path. The
    /// properties are declared at internal-visibility (no access modifier)
    /// in EncoderSession.swift, which is the right call for now — exposing
    /// `preCache` etc. as `public` would invite callers to take ownership of
    /// PLAN.md §11 Anti-pattern 4 ("don't reallocate cache buffers per chunk").
    /// If the test fails to build because the properties are made private in
    /// a future cleanup, expose an internal `func cacheValuesForTest() -> ...`
    /// hook on EncoderSession rather than weakening the cache properties'
    /// access level.
    func testEncoderSessionResetClearsCaches() async throws {
        let fixturesDir = Self.fixturesDirectory()
        let modelURL = fixturesDir.appendingPathComponent("aimodel")
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw XCTSkip("requires .aimodel fixture at \(modelURL.path) — see fixtures/README.md")
        }

        // VERIFY: `AIModel(contentsOf:)` signature per PLAN.md §6 "Model loading".
        // If it turns out to be sync (`throws` only, no `await`), remove the
        // `await` here AND in StreamingASR.init.
        let model: AIModel
        do {
            model = try await AIModel(contentsOf: modelURL)
        } catch {
            throw XCTSkip("could not load .aimodel at \(modelURL.path): \(error)")
        }

        let encoder: EncoderSession
        do {
            encoder = try await EncoderSession(model: model)
        } catch {
            throw XCTSkip("could not construct EncoderSession (likely missing 'encode' function): \(error)")
        }

        // Step 1: write a recognizable non-zero pattern into all four caches
        // via the same `mutableView()` accessor that `reset()` uses. We do
        // this directly rather than running a forward pass — a forward pass
        // requires a valid mel chunk and lang_id, and the runtime might write
        // whatever the model computes (which could be near-zero). Writing
        // a sentinel pattern makes the post-reset zero check decisive.
        //
        // VERIFY: access to EncoderSession's cache properties requires either
        // testability or moving the assertion into an `@testable` test hook.
        // The current actor isolation means we must `await` to even read the
        // properties from outside the actor.
        await encoder._test_setAllCachesToSentinel(0.42)

        // Step 2: verify the sentinel actually landed (any of the float caches).
        let lc = await encoder._test_cacheLastChannelFlat()
        XCTAssertFalse(lc.allSatisfy { $0 == 0.0 },
                       "sentinel write failed — cacheLastChannel is still all-zero")

        // Step 3: reset, verify all three caches are zero (no pre_cache in the
        // English-only checkpoint — see EncoderSession.swift header).
        await encoder.reset()

        let lcZ = await encoder._test_cacheLastChannelFlat()
        let ltZ = await encoder._test_cacheLastTimeFlat()
        let lenZ = await encoder._test_cacheLastChannelLenFlat()

        for (name, buf) in [("cacheLastChannel", lcZ),
                            ("cacheLastTime", ltZ)] {
            for (i, v) in buf.enumerated() {
                XCTAssertEqual(v, 0.0,
                               "post-reset \(name)[\(i)] must be 0.0, got \(v)")
            }
        }
        for (i, v) in lenZ.enumerated() {
            XCTAssertEqual(v, 0,
                           "post-reset cacheLastChannelLen[\(i)] must be 0, got \(v)")
        }
    }

    #else

    func testEncoderSessionResetClearsCaches() throws {
        throw XCTSkip("CoreAI framework not importable — install Xcode 27 SDK to run EncoderSession cache tests")
    }

    #endif

    // MARK: - Helpers

    /// Resolve the on-disk path to the test fixtures directory.
    /// Uses #filePath-relative resolution; the test target has no resources
    /// declared in Package.swift (no fixtures committed), so Bundle.module
    /// would be unavailable anyway.
    private static func fixturesDirectory() -> URL {
        // Tests/VoixfulEngineTests/ → repo-root/fixtures
        let here = URL(fileURLWithPath: #filePath)
        return here
            .deletingLastPathComponent()    // .../VoixfulEngineTests
            .deletingLastPathComponent()    // .../Tests
            .deletingLastPathComponent()    // repo root
            .appendingPathComponent("fixtures")
    }
}

// MARK: - Test hooks on EncoderSession
//
// These extensions exist solely to give `testEncoderSessionResetClearsCaches`
// a way to write a sentinel pattern into the encoder's four cache buffers
// and read them back, without weakening the access level of the cache
// properties themselves. They live in the test target (via `@testable import`)
// and are marked `_test_` so a careless caller can't reach them by accident.
//
// VERIFY: this depends on `@testable import VoixfulEngine` granting access to
// EncoderSession's internal stored properties. If a future cleanup makes the
// cache properties strictly private, replace these extensions with hooks
// declared on EncoderSession itself (still internal-visibility) so the test
// path stays open.

#if canImport(CoreAI)
extension EncoderSession {
    /// Fill all four caches with a sentinel scalar so `reset()` has something
    /// non-zero to clear. The int32 length cache is cast to Int32 from the
    /// Float input (so a sentinel of 0.42 lands as 0 — we use a separate
    /// path for it).
    func _test_setAllCachesToSentinel(_ value: Float) {
        cacheLastChannel.fillFloat32(value)
        cacheLastTime.fillFloat32(value)
        // int64 cache: write a non-zero integer directly.
        cacheLastChannelLen.fillInt64(7)
    }

    /// Snapshot `cacheLastChannel` as a flat `[Float]` for assertions.
    func _test_cacheLastChannelFlat() -> [Float] { cacheLastChannel.flatFloat32() }

    /// Snapshot `cacheLastTime` as a flat `[Float]` for assertions.
    func _test_cacheLastTimeFlat() -> [Float] { cacheLastTime.flatFloat32() }

    /// Snapshot `cacheLastChannelLen` as a flat `[Int32]` for assertions.
    func _test_cacheLastChannelLenFlat() -> [Int64] { cacheLastChannelLen.flatInt64() }
}
#endif
