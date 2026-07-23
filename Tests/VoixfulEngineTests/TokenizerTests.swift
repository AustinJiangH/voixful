// Tokenizer round-trip and lang-tag-stripping tests.
//
// Tokenizer is in stub mode until the SentencePiece dependency is wired up,
// so round-trip tests have two shapes:
//
//   1. With a vocab fixture present, the line-per-token best-effort decode
//      path is exercised: it should produce non-empty output for known
//      token ids and apply the SentencePiece `▁` → space substitution.
//      Round-trip from `[Int32]` → text → expectation.
//   2. The `isLangTag(...)` regex check runs unconditionally — it doesn't
//      depend on a loaded vocab. The pattern matches `<en-US>`-style tags.
//
// When the SentencePiece library lands and the stub-mode fallback in
// Tokenizer.swift is deleted, this file's "with vocab" tests will need to be
// rewritten against the real decode path — but the lang-tag tests will
// continue to work as-is.

import XCTest
import Foundation
@testable import VoixfulEngine

final class TokenizerTests: XCTestCase {

    // MARK: - Lang-tag stripping (no fixture)

    /// `Tokenizer.isLangTag` must accept the canonical `<lang-Region>` form —
    /// two lowercase letters, dash, two uppercase letters, inside angle
    /// brackets. Spot-check a representative locale from each Language tier.
    func testLangTagPatternMatches() {
        XCTAssertTrue(Tokenizer.isLangTag("<en-US>"),
                      "<en-US> must match the lang-tag pattern")
        XCTAssertTrue(Tokenizer.isLangTag("<es-ES>"),
                      "<es-ES> must match")
        XCTAssertTrue(Tokenizer.isLangTag("<zh-CN>"),
                      "<zh-CN> must match")
        XCTAssertTrue(Tokenizer.isLangTag("<ja-JP>"),
                      "<ja-JP> must match")
        XCTAssertTrue(Tokenizer.isLangTag("<mt-MT>"),
                      "<mt-MT> (adaptation-ready tier) must match")
    }

    /// Strings that do NOT match the lang-tag pattern. Anything off-shape —
    /// wrong case, missing brackets, extra characters — must be rejected so
    /// `stripLangTags: true` doesn't accidentally drop legitimate vocabulary.
    func testLangTagPatternRejectsNonTags() {
        XCTAssertFalse(Tokenizer.isLangTag("en-US"),
                       "no angle brackets")
        XCTAssertFalse(Tokenizer.isLangTag("<EN-US>"),
                       "language code must be lowercase")
        XCTAssertFalse(Tokenizer.isLangTag("<en-us>"),
                       "region code must be uppercase")
        XCTAssertFalse(Tokenizer.isLangTag("<en>"),
                       "pattern requires the region suffix — short forms not covered")
        XCTAssertFalse(Tokenizer.isLangTag("<eng-USA>"),
                       "pattern requires 2-letter codes — 3-letter ISO 639-2 not covered")
        XCTAssertFalse(Tokenizer.isLangTag("hello"),
                       "ordinary word must not match")
        XCTAssertFalse(Tokenizer.isLangTag(""),
                       "empty string must not match")
        XCTAssertFalse(Tokenizer.isLangTag("<en-US"),
                       "missing closing bracket")
        XCTAssertFalse(Tokenizer.isLangTag("en-US>"),
                       "missing opening bracket")
        XCTAssertFalse(Tokenizer.isLangTag("<en-US>x"),
                       "trailing junk must not be tolerated")
    }

    // MARK: - Round-trip via stub path

    /// Round-trip a synthetic [Int32] token sequence through the stub
    /// decode path with a minimal fixture vocab. The fixture vocab is a
    /// plain-text file (one piece per line, line number = id) and exercises
    /// the line-per-token loader in `Tokenizer.init`.
    ///
    /// Skipped if the fixture is unavailable. This is `vocab.txt` per
    /// fixtures/README.md.
    func testDecodeRoundTripWithFixtureVocab() async throws {
        let vocabURL = try Self.requireFixture(named: "vocab.txt")

        let tokenizer = try Tokenizer(vocabURL: vocabURL)

        // Pull a few token ids from the vocab itself so the test is robust
        // to vocab-file changes. We read the file we just gave the tokenizer,
        // pick the first three non-empty lines, and assert their pieces
        // appear in the decoded output (modulo SentencePiece `▁` → space
        // substitution).
        let raw = try String(contentsOf: vocabURL, encoding: .utf8)
        let lines = raw.split(separator: "\n", omittingEmptySubsequences: false)
        var idsToTest: [Int32] = []
        var expectedPieces: [String] = []
        for (idx, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }
            // Take the substring up to the first whitespace as the piece,
            // matching Tokenizer.init's parser.
            let piece: String
            if let ws = trimmed.firstIndex(where: { $0 == "\t" || $0 == " " }) {
                piece = String(trimmed[trimmed.startIndex..<ws])
            } else {
                piece = trimmed
            }
            if piece.isEmpty { continue }
            // Skip the blank id (13087) and any lang-tag pieces — they're
            // tested separately.
            if Int32(idx) == GreedyRNNTDecoder.blankId { continue }
            if Tokenizer.isLangTag(piece) { continue }
            idsToTest.append(Int32(idx))
            expectedPieces.append(piece)
            if idsToTest.count == 3 { break }
        }
        guard !idsToTest.isEmpty else {
            throw XCTSkip("vocab fixture at \(vocabURL.path) had no usable non-blank, non-langtag tokens")
        }

        let decoded = await tokenizer.decode(tokens: idsToTest, stripLangTags: false)

        // Stub-mode decode replaces `▁` (U+2581) with a space; the rest of
        // each piece must appear verbatim somewhere in the decoded text.
        for piece in expectedPieces {
            let translated = piece.replacingOccurrences(of: "\u{2581}", with: " ")
            XCTAssertTrue(
                decoded.contains(translated),
                "decoded text \"\(decoded)\" did not contain expected piece \"\(translated)\" (from raw \"\(piece)\")"
            )
        }
    }

    /// `stripLangTags: true` removes any token whose piece matches the
    /// `<lang-Region>` form. Construct a synthetic token sequence that
    /// contains the lang-tag at a known id (looked up by scanning the vocab
    /// for any lang-tag piece). Skip if the vocab contains no lang-tags.
    func testDecodeStripsLangTagsWhenRequested() async throws {
        let vocabURL = try Self.requireFixture(named: "vocab.txt")

        let raw = try String(contentsOf: vocabURL, encoding: .utf8)
        let lines = raw.split(separator: "\n", omittingEmptySubsequences: false)

        var langTagID: Int32?
        var contentID: Int32?
        var contentPiece: String?
        for (idx, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }
            let piece: String
            if let ws = trimmed.firstIndex(where: { $0 == "\t" || $0 == " " }) {
                piece = String(trimmed[trimmed.startIndex..<ws])
            } else {
                piece = trimmed
            }
            if Tokenizer.isLangTag(piece) {
                if langTagID == nil { langTagID = Int32(idx) }
            } else if !piece.isEmpty,
                      Int32(idx) != GreedyRNNTDecoder.blankId,
                      contentID == nil {
                contentID = Int32(idx)
                contentPiece = piece
            }
            if langTagID != nil && contentID != nil { break }
        }
        guard let tag = langTagID, let content = contentID, let piece = contentPiece else {
            throw XCTSkip("vocab fixture at \(vocabURL.path) lacks both a lang-tag and a content piece — cannot exercise strip path")
        }

        let tokenizer = try Tokenizer(vocabURL: vocabURL)
        let withTag = await tokenizer.decode(tokens: [tag, content], stripLangTags: false)
        let withoutTag = await tokenizer.decode(tokens: [tag, content], stripLangTags: true)

        // The translated content piece must appear in both outputs.
        let translatedPiece = piece.replacingOccurrences(of: "\u{2581}", with: " ")
        XCTAssertTrue(withTag.contains(translatedPiece),
                      "lang-tag-included decode should still produce the content piece")
        XCTAssertTrue(withoutTag.contains(translatedPiece),
                      "lang-tag-stripped decode should still produce the content piece")

        // The lang-tag content itself must NOT appear in the strip-true output.
        // We don't know which exact tag string the vocab uses, so check that
        // the stripped output is a prefix-or-substring of the unstripped — i.e.,
        // strictly fewer characters because the tag was removed.
        XCTAssertLessThan(withoutTag.count, withTag.count,
                          "stripLangTags: true should shorten the decoded text by the tag's contribution")
    }

    // MARK: - Helpers

    /// Resolve a fixture file relative to the source tree. Throws `XCTSkip` if
    /// missing — the test target never asserts on a missing fixture.
    private static func requireFixture(named name: String) throws -> URL {
        // #filePath-relative resolution; the test target has no resources
        // declared in Package.swift (no fixtures committed), so Bundle.module
        // would be unavailable anyway.
        let here = URL(fileURLWithPath: #filePath)
        let fixtures = here
            .deletingLastPathComponent()    // .../VoixfulEngineTests
            .deletingLastPathComponent()    // .../Tests
            .deletingLastPathComponent()    // repo root
            .appendingPathComponent("fixtures")
        let url = fixtures.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("fixture \(name) absent at \(url.path) — see fixtures/README.md")
        }
        return url
    }
}
