// Phase K — In-XCTest smoke harness.
//
// Runs Steps 1-3 of the model-free smoke (enum sanity, MelExtractor on
// synthetic audio, LangID.oneHot) on every `swift test` invocation. Step 4
// (full pipeline) is exercised only by the CLI subcommand and only when a
// `--model` path is supplied, since it requires real model artifacts.
//
// The step implementations live in `VoixfulEngine.Smoke` (Sources/VoixfulEngine/Smoke.swift)
// so this test and the `smoke` CLI subcommand exercise the same code.
// Regressions caught here are caught at the CLI level too, and vice-versa.

import XCTest
@testable import VoixfulEngine

final class SmokeTests: XCTestCase {

    /// Step 1: enum-cardinality invariants. `ChunkSize.allCases.count == 5`
    /// and `Language.allCases.count == 40` per PLAN.md §2.
    func testStep1EnumSanity() {
        let result = Smoke.step1EnumSanity()
        XCTAssertEqual(result.status, .pass,
                       "step 1 (enum sanity) failed: \(result.detail)")
    }

    /// Step 2: MelExtractor produces a non-empty, all-finite `[128 × T]` array
    /// from 1 second of synthetic 16 kHz audio. Catches FFT / Welford /
    /// mel-projection regressions that would surface as NaN/Inf or shape drift.
    func testStep2MelExtractor() async {
        let result = await Smoke.step2MelExtractor()
        XCTAssertEqual(result.status, .pass,
                       "step 2 (mel extractor) failed: \(result.detail)")
    }

    /// Step 3: `LangID.oneHot(.enUS)` returns a 128-dim NDArray with exactly
    /// one nonzero slot. On environments without Core AI on the import path
    /// (pre-Xcode-27 toolchain), this step reports `.skipped` rather than
    /// failing — we don't want a half-installed env to flag as a regression.
    func testStep3LangIDOneHot() throws {
        let result = Smoke.step3LangIDOneHot(.enUS)
        switch result.status {
        case .pass:
            break  // happy path
        case .skipped:
            throw XCTSkip(result.detail)
        case .fail:
            XCTFail("step 3 (LangID.oneHot) failed: \(result.detail)")
        }
    }

    /// Run all three model-free steps in one pass and confirm none failed.
    /// Mirrors what the `smoke` CLI subcommand does (minus the full-pipeline
    /// step 4, which requires real model artifacts).
    func testAllStepsTogether() async {
        let r1 = Smoke.step1EnumSanity()
        let r2 = await Smoke.step2MelExtractor()
        let r3 = Smoke.step3LangIDOneHot(.enUS)

        // Step 3 may report `.skipped` on a pre-Xcode-27 toolchain; that's
        // not a failure. Steps 1 and 2 are unconditional.
        XCTAssertNotEqual(r1.status, .fail, "step 1: \(r1.detail)")
        XCTAssertNotEqual(r2.status, .fail, "step 2: \(r2.detail)")
        XCTAssertNotEqual(r3.status, .fail, "step 3: \(r3.detail)")
    }

    /// Sanity-check the audio synthesis helper. Catches off-by-one bugs in
    /// the silence/tone/silence segmentation that would otherwise only
    /// manifest as a shape mismatch in step 2.
    func testSynthesizedAudioShape() {
        let samples = Smoke.synthesizeOneSecondAudio()
        XCTAssertEqual(samples.count, 16_000,
                       "1 s of 16 kHz audio must have 16,000 samples (got \(samples.count))")

        // First 6,400 samples (0.4 s) must be silence.
        for i in 0..<6_400 {
            XCTAssertEqual(samples[i], 0.0, "leading silence at index \(i) was \(samples[i])")
        }
        // Trailing 6,400 samples (0.4 s) must be silence.
        for i in 9_600..<16_000 {
            XCTAssertEqual(samples[i], 0.0, "trailing silence at index \(i) was \(samples[i])")
        }
        // Middle 3,200 samples (0.2 s) must hold the sine. Sample at the
        // quarter-period mark of a 440 Hz tone is the amplitude (approx); we
        // just check non-zero presence rather than reproducing the math.
        var middleNonzeroCount = 0
        for i in 6_400..<9_600 {
            if samples[i] != 0 { middleNonzeroCount += 1 }
        }
        XCTAssertGreaterThan(middleNonzeroCount, 3_000,
                             "expected ~3,200 nonzero tone samples in [6400, 9600); got \(middleNonzeroCount)")
    }

    /// Confirm the `smokeReport(...)` formatter produces the documented
    /// `[smoke] step N: STATUS detail` shape — the CLI driver and `scripts/smoke.sh`
    /// grep on this prefix.
    func testSmokeReportFormat() {
        let pass = smokeReport(step: "1", result: .pass("ok"))
        XCTAssertEqual(pass, "[smoke] step 1: PASS ok")

        let fail = smokeReport(step: "2", result: .fail("bad"))
        XCTAssertEqual(fail, "[smoke] step 2: FAIL bad")

        let skip = smokeReport(step: "3", result: .skipped("no SDK"))
        XCTAssertEqual(skip, "[smoke] step 3: SKIPPED no SDK")
    }
}
