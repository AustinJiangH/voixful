// SentencePiece detokenizer for Cohere Transcribe (16,384-piece vocab).
//
// Handles the three piece kinds in the vocab:
//   - `▁`-prefixed word pieces  -> `▁` becomes a space (SentencePiece convention)
//   - byte-fallback `<0xNN>`     -> the raw byte NN (multi-byte UTF-8 / CJK)
//   - special tokens             -> skipped (`<unk>`, `<pad>`, `<|...|>` task tags)
//
// Bytes from word pieces and byte-fallback tokens accumulate into one buffer so a
// CJK character split across several `<0xNN>` tokens reassembles correctly.

import Foundation

public struct CohereTokenizer {
    private let pieces: [String]   // index == token id

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

    /// True for byte-fallback pieces of the form `<0xNN>`; sets `byte` when so.
    private func byteFallback(_ piece: String) -> UInt8? {
        guard piece.hasPrefix("<0x"), piece.hasSuffix(">"), piece.count == 6 else { return nil }
        let hex = piece.dropFirst(3).dropLast()  // "NN"
        return UInt8(hex, radix: 16)
    }

    /// Detokenize token ids to text (specials skipped, `▁`->space, bytes reassembled).
    public func decode(_ ids: [Int32]) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            let i = Int(id)
            guard i >= 0 && i < pieces.count else { continue }
            let piece = pieces[i]
            if isSpecial(piece) { continue }
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
