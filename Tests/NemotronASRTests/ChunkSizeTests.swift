// Phase H — Unit tests for the `ChunkSize` enum from PLAN.md §2 "Chunk-size dial".
//
// No external fixtures, no Core AI dependency: these tests run unconditionally
// on every `swift test` invocation. They're the cheapest regression guard
// against the §2 chunk-size table drifting away from the published Nemotron 3.5
// checkpoint's `att_context_size` settings.
//
// The five-case invariant (`ChunkSize.allCases.count == 5`) is double-asserted
// in Types.swift's `#if DEBUG _debugAssertCount()` for Language; ChunkSize
// doesn't carry an equivalent guard, so this test is the release-build counterpart.

import XCTest
@testable import NemotronASR

final class ChunkSizeTests: XCTestCase {

    // MARK: - Cardinality

    /// PLAN.md §2 lists exactly 5 chunk-size configurations.
    /// Adding a 6th case (or removing one) requires a matching PLAN.md update.
    func testChunkSizeCount() {
        XCTAssertEqual(ChunkSize.allCases.count, 5,
                       "ChunkSize.allCases.count must be 5 per PLAN.md §2 chunk-size dial")
    }

    // MARK: - Per-case values

    /// `attContextRight` matches the `att_context_size = [56, R]` second value
    /// from PLAN.md §2 for every case. These are constants of the published
    /// checkpoint; if any of these change, the encoder will silently produce
    /// wrong outputs at runtime.
    func testAttContextRight() {
        XCTAssertEqual(ChunkSize.ms80.attContextRight, 0,
                       "ms80 must have att_context_right = 0 (ultra-low latency)")
        XCTAssertEqual(ChunkSize.ms160.attContextRight, 1,
                       "ms160 must have att_context_right = 1 (low latency)")
        XCTAssertEqual(ChunkSize.ms320.attContextRight, 3,
                       "ms320 must have att_context_right = 3 (default — dictation)")
        XCTAssertEqual(ChunkSize.ms560.attContextRight, 6,
                       "ms560 must have att_context_right = 6 (quality-leaning)")
        XCTAssertEqual(ChunkSize.ms1120.attContextRight, 13,
                       "ms1120 must have att_context_right = 13 (near-offline)")
    }

    /// `totalFrames` is `1 + attContextRight` per the §2 table — the chunk
    /// itself is always one 80 ms encoder frame plus the right context.
    func testTotalFrames() {
        XCTAssertEqual(ChunkSize.ms80.totalFrames, 1)
        XCTAssertEqual(ChunkSize.ms160.totalFrames, 2)
        XCTAssertEqual(ChunkSize.ms320.totalFrames, 4)
        XCTAssertEqual(ChunkSize.ms560.totalFrames, 7)
        XCTAssertEqual(ChunkSize.ms1120.totalFrames, 14)
    }

    /// `durationMS` matches the "Total chunk" column of the §2 table verbatim.
    /// Also equals `totalFrames * 80` (encoder frame stride from PLAN.md §2 /
    /// the `// VERIFY: ... 80 ms encoder-frame stride ...` comment in Types.swift).
    func testDurationMS() {
        XCTAssertEqual(ChunkSize.ms80.durationMS, 80)
        XCTAssertEqual(ChunkSize.ms160.durationMS, 160)
        XCTAssertEqual(ChunkSize.ms320.durationMS, 320)
        XCTAssertEqual(ChunkSize.ms560.durationMS, 560)
        XCTAssertEqual(ChunkSize.ms1120.durationMS, 1120)
    }

    /// `durationMS == totalFrames * 80` for every case. This is the derived
    /// invariant that ties the three properties together — if any of them
    /// drifts, this catches it even if the absolute-value tests above are
    /// updated in a refactor.
    func testDurationMatchesFrameCount() {
        for chunk in ChunkSize.allCases {
            XCTAssertEqual(
                chunk.durationMS,
                chunk.totalFrames * 80,
                "ChunkSize.\(chunk) durationMS (\(chunk.durationMS)) must equal totalFrames (\(chunk.totalFrames)) * 80 ms"
            )
        }
    }

    /// `totalFrames == 1 + attContextRight` per the §2 table derivation.
    func testTotalFramesMatchesContext() {
        for chunk in ChunkSize.allCases {
            XCTAssertEqual(
                chunk.totalFrames,
                1 + chunk.attContextRight,
                "ChunkSize.\(chunk) totalFrames (\(chunk.totalFrames)) must equal 1 + attContextRight (\(chunk.attContextRight))"
            )
        }
    }

    // MARK: - Raw values

    /// `rawValue` is the CLI-friendly tag (`"80ms"`, `"320ms"`, etc.).
    /// PLAN.md doesn't pin a specific spelling but the CLI parser in
    /// `NemotronASRCLI` consumes these verbatim.
    func testRawValues() {
        XCTAssertEqual(ChunkSize.ms80.rawValue, "80ms")
        XCTAssertEqual(ChunkSize.ms160.rawValue, "160ms")
        XCTAssertEqual(ChunkSize.ms320.rawValue, "320ms")
        XCTAssertEqual(ChunkSize.ms560.rawValue, "560ms")
        XCTAssertEqual(ChunkSize.ms1120.rawValue, "1120ms")
    }

    /// Round-trip through `rawValue` and `init?(rawValue:)`.
    func testRawValueRoundTrip() {
        for chunk in ChunkSize.allCases {
            XCTAssertEqual(ChunkSize(rawValue: chunk.rawValue), chunk,
                           "ChunkSize.\(chunk).rawValue must round-trip through init?(rawValue:)")
        }
    }
}
