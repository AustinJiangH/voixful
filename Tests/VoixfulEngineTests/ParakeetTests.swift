// Parakeet backend tests — the model-free surface. The TDT greedy decode itself
// needs a real .aimodel (covered by the on-device runs), but the Parakeet-specific
// mel configuration — `torch.stft(center=True)` window centering via the 56-sample
// `frameStartOffset` — is verifiable here.

import XCTest
@testable import VoixfulEngine

final class ParakeetTests: XCTestCase {

    /// Parakeet centers the 400-sample Hann in the 512-sample FFT frame, i.e.
    /// `frameStartOffset = (512 - 400) / 2 = 56`, whereas the Nemotron path uses 0.
    /// The shift must actually change the extracted features (and stay finite).
    func testFrameStartOffsetCentersWindow() async {
        let samples = (0..<16_000).map { Float(sin(Double($0) * 0.07)) }   // 1 s tone
        let leftAligned = await MelExtractor().extract(samples: samples)               // offset 0
        let centered = await MelExtractor(frameStartOffset: 56).extract(samples: samples) // Parakeet

        XCTAssertFalse(leftAligned.isEmpty)
        XCTAssertFalse(centered.isEmpty)
        XCTAssertTrue(leftAligned.allSatisfy { $0.isFinite }, "left-aligned mel must be finite")
        XCTAssertTrue(centered.allSatisfy { $0.isFinite }, "centered mel must be finite")
        XCTAssertNotEqual(leftAligned, centered,
                          "the 56-sample window centering must change the mel features")
    }

    /// The default extractor (`frameStartOffset = 0`) is unchanged by the option —
    /// guards the Nemotron streaming path against accidental regressions.
    func testDefaultOffsetMatchesExplicitZero() async {
        let samples = (0..<8_000).map { Float(sin(Double($0) * 0.05)) }
        let implicit = await MelExtractor().extract(samples: samples)
        let explicitZero = await MelExtractor(frameStartOffset: 0).extract(samples: samples)
        XCTAssertEqual(implicit, explicitZero)
    }
}
