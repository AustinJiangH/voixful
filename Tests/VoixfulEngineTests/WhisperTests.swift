// Whisper AED backend tests — the host-side pieces verifiable without a model:
// backend detection (the distinctive `whisper_decode` function name), byte-level
// BPE detokenization, and the fixed 30 s single-pass window.

import XCTest
import Foundation
@testable import VoixfulEngine
import VoixfulWhisper

final class WhisperTests: XCTestCase {

    // MARK: - Detection

    /// The distinctive `whisper_decode` entrypoint routes to `.whisper`, and is
    /// checked BEFORE the generic Cohere `decode_*` rule so it isn't shadowed.
    func testWhisperByDecodeEntrypoint() {
        let fns = ModelBackend.AssetFunctions(
            names: ["whisper_encode", "whisper_decode"], encodeHasStates: false)
        XCTAssertEqual(ModelBackend.classify(fns), .whisper)
    }

    /// An asset carrying BOTH a Whisper decode and a Cohere-style `decode_step`
    /// (defensive: should never happen) still classifies as Whisper because the
    /// Whisper rule comes first.
    func testWhisperRuleNotShadowedByCohere() {
        let fns = ModelBackend.AssetFunctions(
            names: ["whisper_encode", "whisper_decode", "decode_step"], encodeHasStates: false)
        XCTAssertEqual(ModelBackend.classify(fns), .whisper)
    }

    /// Whisper advertises the fixed 30 s single-pass window.
    func testSinglePassWindow() {
        XCTAssertEqual(ModelBackend.whisper.singlePassWindowSeconds, 30)
    }

    // MARK: - Language prompt token resolution

    // Real Whisper ids: eot=50257, <|en|>=50259. A language token must sit
    // strictly above eot; anything else (absent language, or a corrupt manifest
    // that aliased a language to eot) must fall back to English so the decoder
    // isn't primed with eot and stopped instantly.
    func testLangTokenResolvesPresentLanguage() {
        let t = WhisperEngine.resolveLangToken(
            langs: ["en": 50259, "fr": 50265], lang: "fr", englishDefault: 50259, eot: 50257)
        XCTAssertEqual(t, 50265)
    }

    func testLangTokenFallsBackWhenAbsent() {
        let t = WhisperEngine.resolveLangToken(
            langs: ["en": 50259], lang: "yue", englishDefault: 50259, eot: 50257)
        XCTAssertEqual(t, 50259)
    }

    func testLangTokenFallsBackWhenAliasedToEot() {
        // The shipped bug: <|yue|> mapped to eot (50257). Must not prime with it.
        let t = WhisperEngine.resolveLangToken(
            langs: ["en": 50259, "yue": 50257], lang: "yue", englishDefault: 50259, eot: 50257)
        XCTAssertEqual(t, 50259)
    }

    func testLangTokenFallsBackWhenBelowEot() {
        let t = WhisperEngine.resolveLangToken(
            langs: ["en": 50259, "xx": 12345], lang: "xx", englishDefault: 50259, eot: 50257)
        XCTAssertEqual(t, 50259)
    }

    // MARK: - Byte-level BPE detokenization

    /// Write a temp tokens.txt (one byte-unicode piece per line, line index ==
    /// id) and return a tokenizer over it. Mirrors the real bundled format.
    private func tokenizer(_ pieces: [String], specialStart: Int) throws -> WhisperTokenizer {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("whisper-tokens-\(UUID().uuidString).txt")
        try pieces.joined(separator: "\n").write(to: url, atomically: true, encoding: .utf8)
        return try WhisperTokenizer(tokensURL: url, specialTokenStart: specialStart)
    }

    /// GPT-2 byte-level BPE: 'Ġ' (U+0120) is the byte-encoded space, and ASCII
    /// letters map to themselves. `Ġthe` -> " the".
    func testSpaceMarkerAndAscii() throws {
        let tok = try tokenizer(["\u{0120}the", "\u{0120}cat"], specialStart: 100)
        XCTAssertEqual(tok.decode([0, 1]), " the cat")
    }

    /// Multi-byte UTF-8 split across byte tokens reassembles: "é" = 0xC3 0xA9.
    /// In the byte<->unicode map 0xC3 -> 'Ã' (U+00C3, direct) and 0xA9 -> ' Đ'…
    /// specifically 0xA9 is NOT in the direct printable set so it maps to a
    /// 256+n scalar. We encode the two source bytes the way the convert step
    /// would: each token holds the byte-unicode char for one raw byte.
    func testMultiByteReassembly() throws {
        // byteDecoder: 0xC3 (195) is in 0xAE..0xFF? No — 0xC3 is in 0xAE..0xFF,
        // so it's a DIRECT byte -> scalar U+00C3 ('Ã'). 0xA9 (169) is in
        // 0xA1..0xAC -> direct -> scalar U+00A9 ('©').
        let tok = try tokenizer(["\u{00C3}\u{00A9}"], specialStart: 100)
        XCTAssertEqual(tok.decode([0]), "é")
    }

    /// Special/timestamp tokens (id >= specialStart) are skipped on decode.
    func testSpecialTokensSkipped() throws {
        // ids: 0='Ġhi', 1='<|startoftranscript|>' (>= specialStart), 2='Ġthere'
        let tok = try tokenizer(
            ["\u{0120}hi", "<|startoftranscript|>", "\u{0120}there"], specialStart: 1)
        // Only id 0 is a text token (1 and 2 are >= specialStart=1 -> skipped).
        XCTAssertEqual(tok.decode([0, 1, 2]), " hi")
    }

    /// Out-of-range ids are ignored rather than crashing.
    func testOutOfRangeIdsIgnored() throws {
        let tok = try tokenizer(["\u{0120}ok"], specialStart: 100)
        XCTAssertEqual(tok.decode([0, 999, -5]), " ok")
    }

    /// `count` reports the loaded vocab size (the engine validates it vs the model).
    func testTokenizerCount() throws {
        XCTAssertEqual(try tokenizer(["a", "b", "c"], specialStart: 100).count, 3)
    }

    // MARK: - Log-mel front-end

    /// The front-end emits a fixed `[nMels × melFrames]` regardless of input
    /// length (Whisper pads/trims to the 30 s window). A tiny scaled-down config
    /// keeps the test fast while exercising the same code path.
    func testMelOutputShapeIsFixed() {
        let mel = WhisperMel(sampleRate: 16_000, nMels: 8, nFFT: 400, hop: 160, chunkSamples: 3200)
        // melFrames = chunkSamples / hop = 20.
        XCTAssertEqual(mel.logMel(samples: [Float](repeating: 0, count: 100)).count, 8 * 20)
        XCTAssertEqual(mel.logMel(samples: [Float](repeating: 0.1, count: 99_999)).count, 8 * 20)
    }

    /// Pure silence: power = 0 everywhere -> log10(clamp(0, 1e-10)) = -10 -> the
    /// dynamic-range floor (max - 8) is also -10 -> (x + 4)/4 = -1.5. So a silent
    /// clip normalizes to a constant -1.5, a deterministic check of the whole
    /// log10 / clamp / (x+4)/4 chain.
    func testMelSilenceNormalizesToConstant() {
        let mel = WhisperMel(sampleRate: 16_000, nMels: 8, nFFT: 400, hop: 160, chunkSamples: 3200)
        let out = mel.logMel(samples: [Float](repeating: 0, count: 3200))
        for v in out { XCTAssertEqual(v, -1.5, accuracy: 1e-4) }
    }
}
