// Tests for the shared `perFeatureNormalize` — the NeMo `normalize: per_feature`
// preprocessor step used by the Parakeet and Cohere encoders.

import XCTest
@testable import VoixfulEngine

final class NormalizationTests: XCTestCase {

    /// Single bin: each value standardized to zero mean, unit (unbiased) std.
    func testSingleBinZeroMeanUnitStd() {
        // mel = [1,2,3,4,5]: mean 3, unbiased std = sqrt(10/4) ≈ 1.58114.
        let out = perFeatureNormalize([1, 2, 3, 4, 5], nMels: 1, frames: 5)
        XCTAssertEqual(out[0], -1.26491, accuracy: 1e-4)
        XCTAssertEqual(out[2], 0.0, accuracy: 1e-4)
        XCTAssertEqual(out[4], 1.26491, accuracy: 1e-4)
        let mean = out.reduce(0, +) / Float(out.count)
        XCTAssertEqual(mean, 0.0, accuracy: 1e-5, "normalized bin must be zero-mean")
    }

    /// Each mel bin is normalized independently (row-major [nMels × frames]).
    func testBinsNormalizedIndependently() {
        // bin0 = [10, 20] (mean 15), bin1 = [-5, 5] (mean 0); both unbiased std = sqrt(50).
        let out = perFeatureNormalize([10, 20, -5, 5], nMels: 2, frames: 2)
        XCTAssertEqual(out[0], -0.70711, accuracy: 1e-4)   // bin0
        XCTAssertEqual(out[1], 0.70711, accuracy: 1e-4)
        XCTAssertEqual(out[2], -0.70711, accuracy: 1e-4)   // bin1, different mean, same shape
        XCTAssertEqual(out[3], 0.70711, accuracy: 1e-4)
    }

    /// With fewer than 2 frames the variance is undefined — pass through unchanged.
    func testSingleFramePassthrough() {
        let input: [Float] = [42, 7]   // nMels 2, frames 1
        XCTAssertEqual(perFeatureNormalize(input, nMels: 2, frames: 1), input)
    }
}
