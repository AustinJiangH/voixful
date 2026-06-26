// Cohere AED backend tests — the host-side pieces that are verifiable without a
// model: SentencePiece + byte-fallback detokenization, and the encoder-length
// subsampling formula.

import XCTest
import Foundation
@testable import VoixfulEngine

final class CohereTests: XCTestCase {

    /// Write a temp pieces file (one piece per line, line index == token id) and
    /// return a tokenizer over it. Mirrors the real `cohere-pieces.txt` format.
    private func tokenizer(_ pieces: [String]) throws -> SentencePieceTokenizer {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("cohere-pieces-\(UUID().uuidString).txt")
        try pieces.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
        return try SentencePieceTokenizer(piecesURL: url)
    }

    // MARK: - Detokenization

    /// `▁` (U+2581) is the SentencePiece word boundary and must become a space.
    func testWordPiecesBecomeSpaces() throws {
        let tok = try tokenizer(["<unk>", "\u{2581}Hello", "\u{2581}world", "!"])
        // ids: 1 = ▁Hello, 2 = ▁world, 3 = !
        XCTAssertEqual(tok.decode([1, 2, 3]), " Hello world!")
    }

    /// Special tokens (`<|…|>`, `<unk>`, `<pad>`) are skipped on decode.
    func testSpecialTokensSkipped() throws {
        let tok = try tokenizer([
            "<unk>", "<pad>", "<|startoftranscript|>", "<|en|>", "\u{2581}Hi",
        ])
        // Feed the specials interleaved with one real word.
        XCTAssertEqual(tok.decode([0, 2, 3, 4, 1]), " Hi")
    }

    /// Byte-fallback `<0xNN>` tokens reassemble into multi-byte UTF-8 (e.g. CJK
    /// or accented characters spanning several byte tokens).
    func testByteFallbackReassembly() throws {
        // "é" is U+00E9 = 0xC3 0xA9 in UTF-8.
        let tok = try tokenizer(["<unk>", "<0xC3>", "<0xA9>"])
        XCTAssertEqual(tok.decode([1, 2]), "é")
    }

    /// Byte fallback mixed with normal pieces in one stream.
    func testMixedByteAndWordPieces() throws {
        // "café": ▁caf + 0xC3 + 0xA9  ("é")
        let tok = try tokenizer(["<unk>", "\u{2581}caf", "<0xC3>", "<0xA9>"])
        XCTAssertEqual(tok.decode([1, 2, 3]), " café")
    }

    /// Out-of-range ids are ignored rather than crashing.
    func testOutOfRangeIdsIgnored() throws {
        let tok = try tokenizer(["<unk>", "\u{2581}ok"])
        XCTAssertEqual(tok.decode([1, 999, -5]), " ok")
    }

    /// `count` reports the loaded vocab size (engines validate it vs the model).
    func testTokenizerCount() throws {
        XCTAssertEqual(try tokenizer(["<unk>", "a", "b", "c"]).count, 4)
    }

    /// `stripLangTags` (used by the multilingual Parakeet path) drops `<xx-YY>`
    /// locale tags while preserving byte-fallback reassembly — the case the old
    /// stub tokenizer got wrong (rendered `<0xNN>` literally).
    func testStripLangTagsWithByteFallback() throws {
        // "<es-ES> café": lang tag + ▁caf + 0xC3 + 0xA9 ("é")
        let tok = try tokenizer(["<unk>", "<es-ES>", "\u{2581}caf", "<0xC3>", "<0xA9>"])
        XCTAssertEqual(tok.decode([1, 2, 3, 4], stripLangTags: true), " café")
        // Without stripping, the lang tag is not a special/byte piece, so it stays.
        XCTAssertEqual(tok.decode([1, 2, 3, 4], stripLangTags: false), "<es-ES> café")
    }

    // MARK: - Encoder length formula

    /// `encLen` mirrors the model's three stride-2 subsampling convs
    /// (`_infer_encoder_lengths_from_raw`). Reference values were taken from the
    /// upstream PyTorch model: 545→69, 1000→125, 3500→438.
    func testEncoderLengthFormula() {
        XCTAssertEqual(CohereTranscribeEngine.encLen(545), 69)
        XCTAssertEqual(CohereTranscribeEngine.encLen(1000), 125)
        XCTAssertEqual(CohereTranscribeEngine.encLen(3500), 438)
    }

    /// Output length is clamped to the encoder's fixed 438-frame window.
    func testEncoderLengthClampedToWindow() {
        XCTAssertLessThanOrEqual(CohereTranscribeEngine.encLen(99999), 438)
        XCTAssertGreaterThanOrEqual(CohereTranscribeEngine.encLen(1), 1)
    }
}
