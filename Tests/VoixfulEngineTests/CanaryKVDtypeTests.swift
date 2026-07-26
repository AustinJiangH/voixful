// Unit tests for CanaryQwenEngine.kvScalarType — the pure dtype-selection
// helper that keeps the Canary-Qwen engine non-breaking across the fp16 KV
// change: new fp16 assets take the fast path; legacy fp32 assets (which
// declare neither an fp16 graph state nor a manifest flag) keep the fp32
// path. No asset needed — this locks in the detection ordering.

import CoreAI
import XCTest
@testable import VoixfulEngine

final class CanaryKVDtypeTests: XCTestCase {
    // Graph descriptor is authoritative and wins over the manifest.
    func testGraphFloat16Wins() {
        XCTAssertEqual(
            CanaryQwenEngine.kvScalarType(graphDtype: .float16, manifestKVDtype: nil), .float16)
        // Even if a manifest disagrees, the graph is the source of truth.
        XCTAssertEqual(
            CanaryQwenEngine.kvScalarType(graphDtype: .float16, manifestKVDtype: "float32"), .float16)
    }

    func testGraphFloat32Wins() {
        XCTAssertEqual(
            CanaryQwenEngine.kvScalarType(graphDtype: .float32, manifestKVDtype: "float16"), .float32)
    }

    // Any non-fp16 graph dtype maps to the legacy fp32 path.
    func testGraphOtherDtypeMapsToFloat32() {
        XCTAssertEqual(
            CanaryQwenEngine.kvScalarType(graphDtype: .bfloat16, manifestKVDtype: nil), .float32)
    }

    // No graph dtype available: fall back to the manifest flag.
    func testManifestFallbackFloat16() {
        XCTAssertEqual(
            CanaryQwenEngine.kvScalarType(graphDtype: nil, manifestKVDtype: "float16"), .float16)
    }

    // Legacy assets: neither a graph dtype nor a manifest flag -> fp32.
    func testDefaultsToFloat32() {
        XCTAssertEqual(
            CanaryQwenEngine.kvScalarType(graphDtype: nil, manifestKVDtype: nil), .float32)
        XCTAssertEqual(
            CanaryQwenEngine.kvScalarType(graphDtype: nil, manifestKVDtype: "float32"), .float32)
    }
}
