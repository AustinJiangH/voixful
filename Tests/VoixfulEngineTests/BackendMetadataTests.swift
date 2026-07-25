// Registry metadata on ModelBackend — no model needed, runs in CI always.

import XCTest
@testable import VoixfulEngine

final class BackendMetadataTests: XCTestCase {
    func testSinglePassWindowSeconds() {
        // Streaming backend has no window.
        XCTAssertNil(ModelBackend.nemotron.singlePassWindowSeconds)

        // Full-context backends: the effective single-pass window (min of
        // encoder window and decoder token cap). Must match each engine's
        // maxAudioSamples / 16 000.
        XCTAssertEqual(ModelBackend.parakeet.singlePassWindowSeconds, 30)
        XCTAssertEqual(ModelBackend.cohere.singlePassWindowSeconds, 15)
        XCTAssertEqual(ModelBackend.granite.singlePassWindowSeconds, 32)
        XCTAssertEqual(ModelBackend.canary.singlePassWindowSeconds, 36)
    }

    func testEveryBackendIsClassified() {
        // A window value (or explicit nil) is defined for every case — a new
        // backend must decide its window rather than crash/default silently.
        for backend in ModelBackend.allCases {
            _ = backend.singlePassWindowSeconds
        }
    }
}
