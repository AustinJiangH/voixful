// Phase K — Shared smoke-test logic.
//
// The CLI's `smoke` subcommand and `Tests/VoixfulEngineTests/SmokeTests.swift`
// both exercise the same end-to-end wiring checks. Centralizing the step
// implementations here means a regression caught at the test level is caught
// at the CLI level (and vice-versa) without re-implementing the audio synthesis
// or the assertions in two places.
//
// A smoke test is the smallest-possible end-to-end exercise of the system that
// confirms "the wiring is sane". It does NOT need real model artifacts to be
// useful — Steps 1-3 here run against synthetic audio and stop short of the
// model load step. Step 4 (full pipeline) is only invoked when a `--model`
// path is supplied by the caller.
//
// The steps deliberately return `SmokeStepResult` instead of throwing so the
// driver can run every step and report a summary, rather than aborting on the
// first failure. The CLI subcommand wraps these results in PASS/FAIL/SKIPPED
// lines on stdout; the XCTest wrapper turns them into XCTAssert calls.

import Foundation
import Accelerate

#if canImport(CoreAI)
import CoreAI
#endif

// MARK: - Result type

/// Outcome of a single smoke step. `SKIPPED` is used for steps that can't be
/// exercised in the current environment (e.g., NDArray scalar accessor not
/// supported in the active SDK) — distinct from FAIL so the driver doesn't
/// exit non-zero just because the environment is half-installed.
public enum SmokeStepStatus: Sendable, Equatable {
    case pass
    case fail
    case skipped
}

/// The outcome of one smoke step, with a human-readable detail string for the
/// reporter. The detail is what gets printed after PASS/FAIL/SKIPPED in the
/// `[smoke] step N: ...` line.
public struct SmokeStepResult: Sendable {
    public let status: SmokeStepStatus
    public let detail: String

    public init(status: SmokeStepStatus, detail: String) {
        self.status = status
        self.detail = detail
    }

    public static func pass(_ detail: String) -> SmokeStepResult {
        SmokeStepResult(status: .pass, detail: detail)
    }

    public static func fail(_ detail: String) -> SmokeStepResult {
        SmokeStepResult(status: .fail, detail: detail)
    }

    public static func skipped(_ detail: String) -> SmokeStepResult {
        SmokeStepResult(status: .skipped, detail: detail)
    }
}

// MARK: - Smoke namespace

/// Stateless namespace holding the four model-free smoke steps. Step 4 (full
/// pipeline) lives on the CLI side because it needs `await StreamingASR(...)`
/// and we don't want this library file to take a soft dependency on the CLI's
/// audio synthesis cadence; the test target invokes Steps 1-3 only.
public enum Smoke {

    // MARK: - Step 1: Enum sanity

    /// Validate that the two enums Phase F locked in still hold their
    /// documented invariants: `ChunkSize.allCases.count == 5` and
    /// `Language.allCases.count == 40`. A drift here means somebody added or
    /// removed a case without updating PLAN.md.
    public static func step1EnumSanity() -> SmokeStepResult {
        let chunkCount = ChunkSize.allCases.count
        let langCount = Language.allCases.count
        if chunkCount != 5 {
            return .fail("ChunkSize.allCases.count == \(chunkCount), expected 5")
        }
        if langCount != 40 {
            return .fail("Language.allCases.count == \(langCount), expected 40")
        }
        return .pass("ChunkSize=5, Language=40")
    }

    // MARK: - Step 2: Synthetic audio + MelExtractor

    /// Build a 1-second synthetic-audio buffer and run it through the mel
    /// extractor. The audio is 0.4 s of silence + 0.2 s of a 440 Hz sine
    /// (amplitude 0.3) + 0.4 s of silence at 16 kHz mono Float. This exercises
    /// the FFT → mel projection → log → Welford-normalize pipeline end-to-end
    /// without any model artifacts.
    ///
    /// PASS criteria:
    ///   - Output non-empty (we have at least one mel frame)
    ///   - All values finite (no NaN/Inf — would indicate a math bug)
    ///   - Shape consistent with `[128 × T]` where T is the expected frame count
    public static func step2MelExtractor() async -> SmokeStepResult {
        let samples = synthesizeOneSecondAudio()
        let mel = MelExtractor()
        let output = await mel.extract(samples: samples)

        if output.isEmpty {
            return .fail("MelExtractor returned empty output (synthesizeOneSecondAudio shape bug?)")
        }

        // Check for non-finite values. NaN/Inf would mean log(0) or a divide
        // by zero in the Welford normalization escaped.
        for (i, v) in output.enumerated() {
            if !v.isFinite {
                return .fail("non-finite mel value at index \(i): \(v)")
            }
        }

        // Validate shape. MelExtractor emits row-major [nMels × T] flattened.
        // It mirrors torch.stft(center=True), which reflect-pads n_fft/2 on each
        // side, so T = sample_count / hop_size + 1 = 16000 / 160 + 1 = 101 for
        // 1 s of audio (NOT (n - window)/hop + 1 — that ignores the centering).
        let nMels = 128
        let expectedT = samples.count / 160 + 1
        let expectedSize = nMels * expectedT
        if output.count != expectedSize {
            return .fail(
                "mel shape mismatch: got \(output.count) elements, expected \(expectedSize) "
                + "([\(nMels) × \(expectedT)])"
            )
        }

        return .pass("[\(nMels) × \(expectedT)] — \(output.count) finite mel values")
    }

    // MARK: - Step 3: LangID.oneHot

    /// Call `LangID.oneHot(parsedLanguage)` and verify the resulting NDArray
    /// has 128 dims with exactly one nonzero. Returns `.skipped` when the
    /// Core AI framework isn't importable in this build (pre-Xcode-27) so the
    /// driver doesn't flag a missing SDK as a failure.
    public static func step3LangIDOneHot(_ language: Language) -> SmokeStepResult {
        #if canImport(CoreAI)
        let oneHot = LangID.oneHot(language)

        // VERIFY: NDArray scalar accessor pattern (PLAN.md §10). If the
        // accessor below is unsupported in this env, the build fails before
        // we get here — the test harness picks it up at compile time. The
        // runtime branch we guard is "step ran but the buffer didn't behave
        // as a 128-dim float32" which would imply an SDK-level shape change.
        let shape = oneHot.shape
        if shape.count != 1 || shape.first != LangID.dim {
            return .fail(
                "LangID.oneHot shape mismatch: got \(shape), expected [\(LangID.dim)]"
            )
        }

        var nonzeroCount = 0
        var hotSlot = -1
        let count = oneHot.elementCount
        oneHot.view(as: Float.self).withUnsafePointer { ptr, _, _ in
            for i in 0..<count {
                if ptr[i] != 0 {
                    nonzeroCount += 1
                    hotSlot = i
                }
            }
        }

        if nonzeroCount != 1 {
            return .fail("LangID.oneHot had \(nonzeroCount) nonzero slots, expected exactly 1")
        }

        return .pass("[\(LangID.dim)] one-hot at slot \(hotSlot) for \(language.rawValue)")
        #else
        return .skipped("CoreAI framework not importable — install Xcode 27 SDK")
        #endif
    }

    // MARK: - Synthetic audio

    /// Generate 1 second of synthetic 16 kHz mono Float audio:
    ///   - 0.4 s of digital silence
    ///   - 0.2 s of a 440 Hz sine at amplitude 0.3
    ///   - 0.4 s of digital silence
    ///
    /// Total 16,000 samples. Public so the CLI's Step 4 (full pipeline) can
    /// reuse the same audio and so SmokeTests can call it directly without
    /// reimplementing the recipe.
    public static func synthesizeOneSecondAudio(sampleRate: Int = 16_000) -> [Float] {
        let silenceFrames = sampleRate * 4 / 10  // 0.4 s
        let toneFrames = sampleRate * 2 / 10     // 0.2 s
        let totalFrames = silenceFrames + toneFrames + silenceFrames
        var out = [Float](repeating: 0, count: totalFrames)

        let freq: Float = 440
        let amplitude: Float = 0.3
        let twoPi: Float = 2 * .pi
        for i in 0..<toneFrames {
            let phase = twoPi * freq * Float(i) / Float(sampleRate)
            out[silenceFrames + i] = amplitude * sin(phase)
        }
        // Trailing silence already zero from `repeating: 0` init.
        return out
    }
}

// MARK: - Reporter

/// Format a `[smoke] step N: STATUS detail` line. Extracted so CLI and tests
/// produce identical strings.
public func smokeReport(step: String, result: SmokeStepResult) -> String {
    let tag: String
    switch result.status {
    case .pass: tag = "PASS"
    case .fail: tag = "FAIL"
    case .skipped: tag = "SKIPPED"
    }
    return "[smoke] step \(step): \(tag) \(result.detail)"
}
