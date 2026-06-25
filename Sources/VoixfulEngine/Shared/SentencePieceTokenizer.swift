// SentencePiece detokenizer shared by the full-context backends (Parakeet TDT,
// Cohere Transcribe). Handles every piece kind a SentencePiece-with-byte-fallback
// vocab emits:
//   - `▁`-prefixed word pieces  -> `▁` becomes a space (SentencePiece convention)
//   - byte-fallback `<0xNN>`     -> the raw byte NN (multi-byte UTF-8 / CJK)
//   - special tokens             -> skipped (`<unk>`, `<pad>`, `<s>`, `</s>`, `<|...|>`)
//   - language tags `<xx-YY>`    -> skipped when `stripLangTags` is set (multilingual
//                                   models like Parakeet TDT v3 prefix a locale tag)
//
// Bytes from word pieces and byte-fallback tokens accumulate into one buffer so a
// CJK character split across several `<0xNN>` tokens reassembles correctly. The
// engines validate `count` against the model's vocab so a wrong pieces.txt fails
// loudly instead of detokenizing to garbage.

import Foundation

public struct SentencePieceTokenizer {
    private let pieces: [String]   // index == token id

    /// Number of pieces loaded (engines check this against the model vocab).
    public var count: Int { pieces.count }

    public init(piecesURL: URL) throws {
        let text = try String(contentsOf: piecesURL, encoding: .utf8)
        // One piece per line, line index == id. Keep every line verbatim (pieces
        // may contain `▁`, `<0x..>`, or raw CJK); don't trim interior content.
        var lines = text.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }  // trailing newline
        self.pieces = lines
    }

    private static let spaceMarker = "\u{2581}"  // ▁

    private func isSpecial(_ piece: String) -> Bool {
        if piece.hasPrefix("<|") { return true }      // <|startoftranscript|>, <|en|>, …
        switch piece {
        case "<unk>", "<pad>", "<s>", "</s>": return true
        default: return false
        }
    }

    /// Language tags like `<en-US>` / `<es-ES>` — two lowercase, dash, two uppercase.
    private static let langTagPattern = #"^<[a-z]{2}-[A-Z]{2}>$"#
    private func isLangTag(_ piece: String) -> Bool {
        piece.range(of: Self.langTagPattern, options: .regularExpression) != nil
    }

    /// True for byte-fallback pieces of the form `<0xNN>`; returns the byte when so.
    private func byteFallback(_ piece: String) -> UInt8? {
        guard piece.hasPrefix("<0x"), piece.hasSuffix(">"), piece.count == 6 else { return nil }
        let hex = piece.dropFirst(3).dropLast()  // "NN"
        return UInt8(hex, radix: 16)
    }

    /// Detokenize token ids to text (specials skipped, `▁`->space, bytes reassembled,
    /// optional locale-tag stripping for multilingual models).
    public func decode(_ ids: [Int32], stripLangTags: Bool = false) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            let i = Int(id)
            guard i >= 0 && i < pieces.count else { continue }
            let piece = pieces[i]
            if isSpecial(piece) { continue }
            if stripLangTags && isLangTag(piece) { continue }
            if let b = byteFallback(piece) {
                bytes.append(b)
                continue
            }
            let text = piece.replacingOccurrences(of: Self.spaceMarker, with: " ")
            bytes.append(contentsOf: Array(text.utf8))
        }
        return String(decoding: bytes, as: UTF8.self)
    }
}
