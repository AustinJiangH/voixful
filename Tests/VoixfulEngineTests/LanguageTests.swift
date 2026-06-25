// Phase H — Unit tests for the `Language` enum and `LangID.oneHot(...)` helper
// from PLAN.md §2 (40 locales) and §3 (128-dim prompt-kernel one-hot).
//
// No external fixtures: these tests run unconditionally on every `swift test`
// invocation. They guard the 40-locale invariant, the BCP-47-style raw-value
// formatting, and the 128-dim one-hot construction. Together they're the
// release-build counterpart to the `#if DEBUG _debugAssertCount()` in
// Types.swift.
//
// The `LangID.oneHot(...)` test does NOT exercise any Core AI model — it only
// reads from the NDArray buffer to verify the one-hot pattern. If `NDArray`
// itself is unavailable at compile time (Core AI framework not installed yet,
// pre-Xcode-27), the corresponding test is compiled out via `#if canImport(CoreAI)`.

import XCTest
@testable import VoixfulEngine

#if canImport(CoreAI)
import CoreAI
#endif

final class LanguageTests: XCTestCase {

    // MARK: - Cardinality

    /// PLAN.md §2 lists exactly 40 supported locales (19 transcription-ready +
    /// 13 broad-coverage + 8 adaptation-ready). Adding or removing a case
    /// without updating PLAN.md is a bug.
    func testLanguageCount() {
        XCTAssertEqual(Language.allCases.count, 40,
                       "Language.allCases.count must be 40 per PLAN.md §2 (got \(Language.allCases.count))")
    }

    // MARK: - Raw value formatting

    /// Spot-check that `rawValue` uses the BCP-47-style `<lang>-<region>` form
    /// PLAN.md §3 documents. The hyphen between language and region (NOT an
    /// underscore) matters because the vocab's `<en-US>`-style lang tags are
    /// matched against `Tokenizer.langTagPattern` (`^<[a-z]{2}-[A-Z]{2}>$`).
    func testLanguageRawValues() {
        XCTAssertEqual(Language.enUS.rawValue, "en-US")
        XCTAssertEqual(Language.enGB.rawValue, "en-GB")
        XCTAssertEqual(Language.zhCN.rawValue, "zh-CN")
        XCTAssertEqual(Language.jaJP.rawValue, "ja-JP")
        XCTAssertEqual(Language.deDE.rawValue, "de-DE")
        XCTAssertEqual(Language.nnNO.rawValue, "nn-NO")
        XCTAssertEqual(Language.mtMT.rawValue, "mt-MT")
    }

    /// Every Language's rawValue matches the BCP-47-style pattern
    /// `^[a-z]{2}-[A-Z]{2}$` — two lowercase letters, dash, two uppercase
    /// letters. This is the form Nemotron's vocab tags use (PLAN.md §3) and
    /// the form `Tokenizer.isLangTag(...)` matches against (with the angle
    /// brackets added).
    func testAllLanguagesMatchPattern() {
        let pattern = #"^[a-z]{2}-[A-Z]{2}$"#
        for lang in Language.allCases {
            XCTAssertNotNil(
                lang.rawValue.range(of: pattern, options: .regularExpression),
                "Language.\(lang).rawValue (\"\(lang.rawValue)\") must match BCP-47 form \(pattern)"
            )
        }
    }

    /// All 40 raw values are unique — guards against a copy/paste mistake in
    /// the enum case list. Set-cardinality equals enum-cardinality.
    func testRawValuesAreUnique() {
        let raws = Set(Language.allCases.map { $0.rawValue })
        XCTAssertEqual(raws.count, Language.allCases.count,
                       "Language enum has duplicate rawValues")
    }

    /// Round-trip through `rawValue` and `init?(rawValue:)`.
    func testRawValueRoundTrip() {
        for lang in Language.allCases {
            XCTAssertEqual(Language(rawValue: lang.rawValue), lang,
                           "Language.\(lang).rawValue must round-trip through init?(rawValue:)")
        }
    }

    // MARK: - LangID.oneHot

    #if canImport(CoreAI)

    /// `LangID.oneHot(.enUS)` returns a 128-dim NDArray with slot 0 set to 1.0
    /// and the remaining 127 slots set to 0. Slot 0 because `.enUS` is the
    /// first case in `Language.allCases` and `LangID.oneHot(...)` uses
    /// `allCases.firstIndex(of:)` as the canonical mapping.
    ///
    /// VERIFY: this assumes the NeMo prompt-kernel slot order matches the
    /// Swift `Language` enum declaration order. PLAN.md §10 / LangID.swift
    /// VERIFY (2) flags this as Phase 1 work — once the NeMo lookup table is
    /// confirmed, either reorder the enum or add an explicit mapping.
    func testLangIDOneHot() throws {
        let oneHot = LangID.oneHot(.enUS)

        // VERIFY: NDArray.shape accessor name per PLAN.md §4 — could also be
        // `.dimensions` or unwrap differently. If the SDK exposes it as
        // `[Int]` directly this works; if it's a `Shape` struct, adapt.
        let shape = oneHot.shape
        XCTAssertEqual(shape.count, 1,
                       "LangID.oneHot returns rank-1; PLAN.md §3 hypothesizes [128] (vs [1, 128])")
        XCTAssertEqual(shape.first, LangID.dim,
                       "LangID.oneHot's dim must be \(LangID.dim) (got \(shape.first ?? -1))")
        XCTAssertEqual(LangID.dim, 128, "LangID.dim must equal 128 per PLAN.md §2")

        // Read the buffer and verify the one-hot pattern. Uses the
        // `flatFloat32()` helper from NDArrayHelpers.swift (which wraps the
        // SDK 27.0 `view(as:).withUnsafePointer { ... }` accessor).
        let buf = oneHot.flatFloat32()
        XCTAssertEqual(buf.count, LangID.dim,
                       "buffer count must equal LangID.dim (\(LangID.dim))")
        XCTAssertEqual(buf[0], 1.0,
                       "slot 0 (.enUS) must be 1.0, got \(buf[0])")
        for i in 1..<buf.count {
            XCTAssertEqual(buf[i], 0.0,
                           "slot \(i) must be 0.0, got \(buf[i])")
        }
    }

    /// Spot-check that the hot slot index matches `Language.allCases.firstIndex(of:)`
    /// for a non-first locale (zh-CN, which is in the broad-coverage tier and
    /// thus has a non-trivial slot index).
    func testLangIDOneHotZhCN() throws {
        guard let expectedSlot = Language.allCases.firstIndex(of: .zhCN) else {
            XCTFail("Language.zhCN missing from allCases — should be unreachable")
            return
        }
        let oneHot = LangID.oneHot(.zhCN)
        let buf = oneHot.flatFloat32()
        XCTAssertEqual(buf[expectedSlot], 1.0,
                       "slot \(expectedSlot) (.zhCN) must be 1.0")
        // Spot-check a different slot is 0.
        XCTAssertEqual(buf[0], 0.0,
                       "slot 0 (not .zhCN) must be 0.0")
    }

    #else

    /// `LangID.oneHot` cannot be exercised without the Core AI framework
    /// (it constructs an `NDArray`). When this test bundle is built against
    /// a toolchain that doesn't have Core AI on the import path, this test
    /// reports skip explicitly so the absence is visible — not silently
    /// missing from the report.
    func testLangIDOneHot() throws {
        throw XCTSkip("CoreAI framework not importable — install Xcode 27 SDK to run LangID tests")
    }

    #endif
}
