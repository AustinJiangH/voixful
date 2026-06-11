// Phase E — SentencePiece tokenizer for Nemotron 3.5 ASR.
//
// Detokenizes the int32 token ids emitted by the greedy RNN-T decoder back
// into a UTF-8 string. Nemotron 3.5 ASR uses a SentencePiece BPE vocabulary
// with 13,087 entries (matches blank id; vocab is 0..=13087 with 13087 also
// being blank — the joiner produces logits over the full vocab and blank
// shares the last slot, per PLAN.md §2). The vocabulary includes language
// tags like `<en-US>`, `<es-ES>`, etc., that the prompt-kernel language
// conditioning may surface in output — PLAN.md §3 covers the strip option.
//
// # Contract
//
// - Input: `[Int32]` of token ids, in emit order, as produced by
//   `GreedyRNNTDecoder.decode(...)`. The blank id (13087) does NOT appear in
//   this input (the decoder filters blanks); if it does, it's treated as an
//   unknown token and skipped with a debug-log warning.
// - Output: detokenized `String`. SentencePiece's `▁` (U+2581, "lower one
//   eighth block") prefix becomes a space in the standard `decode()` call;
//   we rely on the underlying SentencePiece library for that.
//
// # SentencePiece dependency
//
// Per the Phase E spec: we do not modify `Package.swift` to add an unverified
// SentencePiece Swift package URL. Instead, the tokenizer is structured around
// a clean interface and currently returns `""` from `decode(...)`, logging a
// one-shot warning so the absence is loud at first run but not noisy. When
// the dependency is wired up, the body of `decode(...)` swaps out for the
// real library call without changing the public API.
//
// See PLAN.md §12 Resources for the candidate libraries to evaluate.

import Foundation
import os.log

public actor Tokenizer {
    // MARK: - Stored state

    /// URL of the vocabulary file passed at init. Retained so that, once the
    /// SentencePiece dependency is wired up, the actual library loader can
    /// consume the same path without a re-init dance.
    public let vocabURL: URL

    /// Token-id → piece-string mapping, built from the vocab file.
    ///
    /// VERIFY: vocab file format. Two candidates from PLAN.md and NeMo
    /// convention:
    ///   1. SentencePiece `.model` (protobuf binary) — would require the
    ///      SentencePiece runtime to load; we can NOT parse this in pure
    ///      Swift without the library, so this code path leaves
    ///      `idToPiece` empty and defers to the library at decode time.
    ///   2. Plain text vocab (one token per line, line number = id) — common
    ///      NeMo export form; trivially parseable. The current loader
    ///      attempts this first and silently leaves the mapping empty on
    ///      parse failure (with a log line).
    /// Phase 1 should inspect what NeMo emits for this checkpoint and lock
    /// the choice in.
    var idToPiece: [Int32: String]

    /// Logger for the one-shot "tokenizer is a stub" warning. We log at most
    /// once per actor instance to avoid spamming the console during the
    /// inevitable Phase F shakeout when nothing produces real text yet.
    let logger = Logger(subsystem: "NemotronASR", category: "Tokenizer")

    /// Set after the first `decode(...)` call so the stub-mode warning fires
    /// exactly once.
    var hasWarnedStubMode: Bool

    /// Regex used by `stripLangTags` to recognize tokens like `<en-US>`,
    /// `<es-ES>`, etc. Per PLAN.md §3 the format is two lowercase + dash +
    /// two uppercase letters inside angle brackets.
    ///
    /// VERIFY: exact lang-tag format in the published vocab. PLAN.md §3
    /// mentions `<en-US>`-style tags; whether the vocab also contains
    /// `<en>` (just the language, no region) or other locale forms is
    /// unconfirmed. Phase 1's vocab dump will tell us.
    static let langTagPattern = #"^<[a-z]{2}-[A-Z]{2}>$"#

    // MARK: - Init

    /// Construct a tokenizer backed by the vocabulary file at `vocabURL`.
    ///
    /// - Parameter vocabURL: Path to the SentencePiece `.model` file (binary)
    ///   or a plain-text vocab (one token per line). Format auto-detected;
    ///   see `idToPiece` doc-comment.
    /// - Throws: `ASRError.modelLoadFailed` if the file cannot be read at all
    ///   (parse failures degrade gracefully — empty mapping + stub-mode
    ///   warning at first decode).
    public init(vocabURL: URL) throws {
        self.vocabURL = vocabURL
        self.hasWarnedStubMode = false

        // Attempt the plain-text path first. If it fails (binary file, IO
        // error, etc.), we leave the mapping empty and rely on the
        // not-yet-wired SentencePiece library at decode time.
        var mapping: [Int32: String] = [:]
        do {
            let contents = try String(contentsOf: vocabURL, encoding: .utf8)
            // VERIFY: vocab file line format. NeMo's plain-text vocab is
            // typically "<piece>\t<score>" or just "<piece>"; we accept both
            // and use the line index as the id. If the actual format is
            // different (e.g., id-first columns), this whole branch needs
            // to be rewritten — but it'll still degrade to the empty-mapping
            // stub path until then.
            for (index, line) in contents.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
                let trimmed = line.drop(while: { $0 == " " })
                if trimmed.isEmpty { continue }
                // Take the substring up to the first whitespace as the piece.
                let piece: String
                if let tabIndex = trimmed.firstIndex(where: { $0 == "\t" || $0 == " " }) {
                    piece = String(trimmed[trimmed.startIndex..<tabIndex])
                } else {
                    piece = String(trimmed)
                }
                mapping[Int32(index)] = piece
            }
        } catch {
            // Couldn't read as UTF-8 text. Probably a binary .model file.
            // Don't throw — the SentencePiece library (once wired up) will
            // handle it directly from `vocabURL`. The stub-mode warning at
            // first decode tells the caller what's up.
        }
        self.idToPiece = mapping
    }

    // MARK: - Decode

    /// Detokenize a sequence of token ids back into a `String`.
    ///
    /// - Parameters:
    ///   - tokens: Token ids in emit order, as produced by
    ///     `GreedyRNNTDecoder.decode(...)`. Should NOT contain the blank id
    ///     (13087); if it does, that entry is skipped.
    ///   - stripLangTags: If true, drop tokens whose pieces match
    ///     `^<[a-z]{2}-[A-Z]{2}>$` (per PLAN.md §3). Useful for surfaces
    ///     that want the raw transcription text without locale markers.
    /// - Returns: Detokenized text. Returns `""` in stub mode (logs a
    ///   one-shot warning on first call); returns a best-effort detokenized
    ///   string from the line-per-token vocab if that path succeeded at init.
    public func decode(tokens: [Int32], stripLangTags: Bool = false) -> String {
        // TODO Phase E follow-up — pick SentencePiece dep at first run.
        // See PLAN.md Resources.
        //
        // When the SentencePiece library is wired up, the body becomes:
        //
        //     let pieces = tokens.compactMap { idToPiece[$0] }
        //     let filtered = stripLangTags ? pieces.filter { !isLangTag($0) } : pieces
        //     return sentencepieceProcessor.decodePieces(filtered)
        //
        // The SentencePiece library handles the `▁` → space conversion and
        // any byte-fallback pieces. We do NOT hand-roll that here because
        // the byte-fallback path is the most-likely-wrong corner of a
        // hand-rolled detokenizer.

        // Stub-mode fallback: if we have a usable piece mapping (line-per-token
        // vocab parsed at init), produce a best-effort string by concatenating
        // pieces with the SentencePiece space-marker translated naively. This
        // is wrong for byte-fallback tokens and for any non-ASCII text that
        // depends on the library's normalization rules, but it gives the
        // Phase F integration something to look at.
        if !idToPiece.isEmpty {
            var pieces: [String] = []
            pieces.reserveCapacity(tokens.count)
            for id in tokens {
                if id == GreedyRNNTDecoder.blankId {
                    // Blank should not appear in decoder output; skip
                    // defensively rather than producing garbage.
                    continue
                }
                guard let piece = idToPiece[id] else {
                    continue
                }
                if stripLangTags && Self.isLangTag(piece) {
                    continue
                }
                pieces.append(piece)
            }
            // SentencePiece convention: pieces starting with `▁` (U+2581) mean
            // "preceded by a space". The library's `decodePieces` handles this
            // correctly; we approximate by replacing `▁` with " " and joining.
            // VERIFY: this naive replacement is wrong for byte-fallback pieces
            // (`<0x20>` etc.) and for the very first piece (where leading `▁`
            // shouldn't produce a leading space). Acceptable as a Phase F
            // smoke-test stub; replace before any WER measurement.
            let joined = pieces.joined()
            return joined.replacingOccurrences(of: "\u{2581}", with: " ")
        }

        // No vocab loaded — emit the one-shot stub-mode warning and return
        // the empty string. The decoder upstream is still exercised; the
        // string-shaped output is just placeholder until the SentencePiece
        // dependency lands.
        if !hasWarnedStubMode {
            logger.warning(
                """
                NemotronASR.Tokenizer is running in stub mode: no SentencePiece \
                dependency wired up and the vocab file at \(self.vocabURL.path, privacy: .public) \
                did not parse as a plain-text per-line vocab. decode(...) will \
                return an empty string until the dependency is added. See PLAN.md \
                §12 Resources and the TODO in Tokenizer.swift.
                """
            )
            hasWarnedStubMode = true
        }
        return ""
    }

    // MARK: - Helpers

    /// Returns `true` if `piece` matches the language-tag pattern from
    /// PLAN.md §3 (e.g., `<en-US>`, `<es-ES>`).
    nonisolated static func isLangTag(_ piece: String) -> Bool {
        // VERIFY: lang tag regex. PLAN.md §3 documents `<en-US>` form; the
        // pattern below assumes exactly two lowercase + dash + two uppercase
        // letters inside angle brackets. If the vocab also has shorter forms
        // (e.g., `<en>`) or longer ones (e.g., `<zh-Hant-TW>`), widen this.
        return piece.range(of: langTagPattern, options: .regularExpression) != nil
    }
}
