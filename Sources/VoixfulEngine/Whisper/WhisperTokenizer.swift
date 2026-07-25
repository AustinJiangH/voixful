// Whisper byte-level BPE detokenizer.
//
// Whisper uses a GPT-2 / tiktoken byte-level BPE vocabulary — NOT SentencePiece,
// so the shared `SentencePieceTokenizer` does not apply. The convert pipeline
// bundles `tokens.txt` inside the .aimodel (one token per line, line index ==
// id) where each line is the token's GPT-2 *byte-to-unicode* representation
// (the same reversible mapping OpenAI/HF use so every token is a printable
// string with no embedded newlines).
//
// Decoding is: for each text token, map each unicode scalar back to its byte via
// the inverse byte<->unicode table, accumulate bytes across tokens, then UTF-8
// decode the whole buffer (so a multi-byte character split over several tokens
// reassembles correctly). Special / control / timestamp tokens (id >=
// `specialTokenStart`) are byte-agnostic markers and are skipped.

import Foundation

struct WhisperTokenizer {
    private let tokens: [String]        // index == token id, byte-unicode form
    private let specialStart: Int       // first special-token id (skip id >= this)
    private let byteForScalar: [UInt32: UInt8]

    /// Number of pieces loaded (the engine checks this against the model vocab).
    var count: Int { tokens.count }

    init(tokensURL: URL, specialTokenStart: Int) throws {
        let text = try String(contentsOf: tokensURL, encoding: .utf8)
        var lines = text.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }  // trailing newline
        self.tokens = lines
        self.specialStart = specialTokenStart
        self.byteForScalar = Self.byteDecoder()
    }

    /// GPT-2 byte<->unicode inverse map (unicode scalar value -> raw byte).
    /// Mirrors `bytes_to_unicode()`: the printable byte ranges map to themselves;
    /// every other byte maps to a distinct scalar at 256+n in byte order.
    private static func byteDecoder() -> [UInt32: UInt8] {
        var direct = Set<Int>()
        for b in 0x21...0x7E { direct.insert(b) }   // '!'..'~'
        for b in 0xA1...0xAC { direct.insert(b) }   // '¡'..'¬'
        for b in 0xAE...0xFF { direct.insert(b) }   // '®'..'ÿ'
        var map: [UInt32: UInt8] = [:]
        for b in direct { map[UInt32(b)] = UInt8(b) }
        var n: UInt32 = 0
        for b in 0...255 where !direct.contains(b) {
            map[256 + n] = UInt8(b)
            n += 1
        }
        return map
    }

    /// Detokenize text token ids to a `String`. Ids >= `specialStart`
    /// (start-of-transcript, language, task, no-timestamps, timestamp, EOT …)
    /// are skipped; the rest map back to raw bytes and UTF-8 decode.
    func decode(_ ids: [Int32]) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            let i = Int(id)
            guard i >= 0, i < tokens.count, i < specialStart else { continue }
            for scalar in tokens[i].unicodeScalars {
                if let b = byteForScalar[scalar.value] { bytes.append(b) }
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}
