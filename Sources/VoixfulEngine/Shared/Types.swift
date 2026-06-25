// Phase D / Phase F — Shared types used across the VoixfulEngine package.
//
// These types intentionally have no Core AI dependency so they can be imported
// by every other file (sessions, decoder, public NemotronStreamingEngine actor, CLI) without
// pulling the framework into modules that don't need it.
//
// Phase F additions: `ChunkSize` (the 5 cache-aware streaming configurations
// from PLAN.md §2 "Chunk-size dial") and `Language` (the 40 locales from PLAN.md
// §2 supported by the published Nemotron 3.5 checkpoint).

import Foundation

// MARK: - Errors

/// Errors thrown by every layer of the VoixfulEngine package.
///
/// Kept `Sendable` so they can cross actor / `Task` boundaries — the public
/// `NemotronStreamingEngine` actor in Phase F propagates these out to callers via `throws`
/// on its `async` methods.
public enum ASRError: Error, Sendable {
    /// `AIModel(contentsOf:)` failed (file missing, corrupt, unsupported
    /// architecture, OS-version mismatch). Associated value is a human-readable
    /// description suitable for logging.
    case modelLoadFailed(String)

    /// `model.loadFunction(named:)` returned `nil` for the named function.
    /// Associated value is the function name we asked for ("encode",
    /// "predict_step", "join").
    case missingFunction(String)

    /// `function.run(...)` returned an `outputs` map that did not contain the
    /// expected key. Associated value is the missing output name.
    case missingOutput(String)

    /// Input shape, dtype, or value out of range. Associated value describes
    /// which input and what was wrong.
    case invalidInput(String)

    /// Audio format unsupported by `feed(...)` (e.g., not 16 kHz mono Float).
    /// Associated value is a description of the offending format.
    case unsupportedAudioFormat(String)
}

// MARK: - Streaming output

/// A streaming partial hypothesis emitted by `NemotronStreamingEngine.feed(...)` as audio
/// chunks are consumed.
///
/// `stableUntilToken` is the index in `text` (token index, not character index)
/// up to which the hypothesis is considered stable — i.e., the right-context
/// window has already passed those tokens and they will not be revised by
/// subsequent chunks. UI surfaces use this to render the stable prefix in one
/// style and the unstable tail in another.
public struct PartialHypothesis: Sendable {
    /// The current best-effort transcription text (detokenized).
    public let text: String

    /// Index into the token sequence behind `text` up to which the hypothesis
    /// is committed (right context already past). Counted in tokens, not
    /// characters or words.
    public let stableUntilToken: Int

    /// Wall-clock timestamp (seconds since session start) at which this
    /// hypothesis was produced. Useful for latency measurement and UI animation.
    public let timestamp: TimeInterval

    public init(text: String, stableUntilToken: Int, timestamp: TimeInterval) {
        self.text = text
        self.stableUntilToken = stableUntilToken
        self.timestamp = timestamp
    }
}

// MARK: - Final transcript

/// The final transcript returned by `NemotronStreamingEngine.finish()` once the audio
/// stream is closed.
public struct Transcript: Sendable {
    /// The full detokenized transcript text, including any whitespace and
    /// punctuation produced by the model.
    public let text: String

    /// Time-aligned segments. For RNN-T greedy decoding we emit one segment
    /// per non-blank token group between blanks (roughly: per word in English),
    /// with start/end times derived from the encoder frame index. Phase F fills
    /// this in; the public shape is fixed now to avoid breaking callers later.
    public let segments: [Segment]

    /// Language tag actually used by the model for this utterance (e.g.,
    /// `"en-US"`). `nil` if language conditioning was disabled or the model
    /// auto-detected and we did not surface the result.
    public let language: String?

    /// One time-aligned segment of the final transcript.
    public struct Segment: Sendable {
        public let text: String
        public let startTime: TimeInterval
        public let endTime: TimeInterval

        public init(text: String, startTime: TimeInterval, endTime: TimeInterval) {
            self.text = text
            self.startTime = startTime
            self.endTime = endTime
        }
    }

    public init(text: String, segments: [Segment], language: String?) {
        self.text = text
        self.segments = segments
        self.language = language
    }
}

// MARK: - Chunk size dial

/// The five cache-aware streaming chunk configurations supported by the
/// published Nemotron 3.5 checkpoint, per PLAN.md §2 ("Chunk-size dial").
///
/// Each case corresponds to a distinct `att_context_size = [56, R]` setting on
/// the encoder, where `R` is the number of right-context 80 ms frames included
/// in the chunk. A single set of weights covers all five at runtime by varying
/// `att_context_size`; whether the runtime selects between them via different
/// inference function calls or via baked-per-`.aimodel` variants is the §10
/// open question Phase 2/3 resolves. Until then, `NemotronStreamingEngine` accepts this
/// enum at init and uses it solely to gate how many mel frames to accumulate
/// before invoking the encoder.
///
/// `rawValue` is the human-readable duration tag suitable for CLI flag values
/// and config files (e.g., `--chunk-size 320ms`).
public enum ChunkSize: String, CaseIterable, Sendable {
    /// 80 ms total chunk, 0 frames of right context. Ultra-low latency
    /// (e.g., voice command). `att_context_size = [56, 0]`.
    case ms80 = "80ms"

    /// 160 ms total chunk, 1 frame (80 ms) of right context. Low latency.
    /// `att_context_size = [56, 1]`.
    case ms160 = "160ms"

    /// 320 ms total chunk, 3 frames (240 ms) of right context. Default —
    /// dictation / captioning. `att_context_size = [56, 3]`.
    case ms320 = "320ms"

    /// 560 ms total chunk, 6 frames (480 ms) of right context.
    /// Quality-leaning streaming. `att_context_size = [56, 6]`.
    case ms560 = "560ms"

    /// 1120 ms total chunk, 13 frames (1040 ms) of right context.
    /// Near-offline. `att_context_size = [56, 13]`.
    case ms1120 = "1120ms"

    /// The right-context size R from `att_context_size = [56, R]` per PLAN.md §2.
    ///
    /// This is the value the encoder needs to know about; the left context (56)
    /// is the same across all five cases and is baked into the cache shapes
    /// (`cache_last_channel` second-to-last axis).
    public var attContextRight: Int {
        switch self {
        case .ms80: return 0
        case .ms160: return 1
        case .ms320: return 3
        case .ms560: return 6
        case .ms1120: return 13
        }
    }

    /// Total mel frames in one chunk = 1 (the "current" 80 ms frame) +
    /// right context frames. This is how many mel frames `NemotronStreamingEngine` must
    /// accumulate before invoking the encoder.
    ///
    /// Derived as `1 + attContextRight`: PLAN.md §2 table shows the chunk
    /// itself is always one 80 ms frame and the variable is right context.
    /// (NeMo's frame rate at the cache-aware FastConformer subsampling output
    /// is 80 ms / frame; the mel preprocessor runs at 10 ms / frame and the
    /// encoder's first subsampling stack folds 8 mel frames → 1 encoder frame.)
    ///
    /// VERIFY: confirm 80 ms encoder-frame stride against PLAN.md §10 open
    /// questions — the mel→encoder-frame factor depends on the subsampling
    /// stack inside `nemo_encoder.streaming_forward(...)` (see
    /// convert/nemotron_convert/wrappers.py). If the stack downsamples by
    /// a different factor, the totalFrames math here moves but the table
    /// values still hold for the encoder-frame-count interpretation.
    public var totalFrames: Int {
        switch self {
        case .ms80: return 1
        case .ms160: return 2
        case .ms320: return 4
        case .ms560: return 7
        case .ms1120: return 14
        }
    }

    /// Total chunk duration in milliseconds — matches the PLAN.md §2 table
    /// and the `rawValue` tag.
    public var durationMS: Int {
        switch self {
        case .ms80: return 80
        case .ms160: return 160
        case .ms320: return 320
        case .ms560: return 560
        case .ms1120: return 1120
        }
    }
}

// MARK: - Language

/// The 40 locales supported by the published Nemotron 3.5 ASR streaming
/// checkpoint, per PLAN.md §2.
///
/// Locales are grouped by support tier on the model card:
///   - 19 transcription-ready (production WER targets met)
///   - 13 broad-coverage (lower-resource locales with reasonable but
///     unevaluated quality)
///   - 8 adaptation-ready (require fine-tuning to reach production quality;
///     ship as-is for experimentation)
///
/// `rawValue` is the BCP-47-ish `<lang>-<region>` tag used by the model's
/// prompt-kernel language conditioning and surfaced verbatim in the
/// `<en-US>`-style language tags inside the vocabulary (PLAN.md §3).
///
/// `allCases.count` must be exactly 40 — this is a sanity check on the
/// language enumeration. The `dim = 128` one-hot vector built by `LangID`
/// allocates 88 unused slots above the 40 supported locales; that's the
/// model's published prompt-kernel dimensionality, not a per-language count
/// (see PLAN.md §2).
public enum Language: String, CaseIterable, Sendable {

    // MARK: - 19 transcription-ready locales

    case enUS = "en-US"
    case enGB = "en-GB"
    case esUS = "es-US"
    case esES = "es-ES"
    case frFR = "fr-FR"
    case frCA = "fr-CA"
    case itIT = "it-IT"
    case ptBR = "pt-BR"
    case ptPT = "pt-PT"
    case nlNL = "nl-NL"
    case deDE = "de-DE"
    case trTR = "tr-TR"
    case ruRU = "ru-RU"
    case arAR = "ar-AR"
    case hiIN = "hi-IN"
    case jaJP = "ja-JP"
    case koKR = "ko-KR"
    case viVN = "vi-VN"
    case ukUA = "uk-UA"

    // MARK: - 13 broad-coverage locales

    case plPL = "pl-PL"
    case svSE = "sv-SE"
    case csCZ = "cs-CZ"
    case nbNO = "nb-NO"
    case daDK = "da-DK"
    case bgBG = "bg-BG"
    case fiFI = "fi-FI"
    case hrHR = "hr-HR"
    case skSK = "sk-SK"
    case zhCN = "zh-CN"
    case huHU = "hu-HU"
    case roRO = "ro-RO"
    case etEE = "et-EE"

    // MARK: - 8 adaptation-ready locales

    case elGR = "el-GR"
    case ltLT = "lt-LT"
    case lvLV = "lv-LV"
    case mtMT = "mt-MT"
    case slSI = "sl-SI"
    case heIL = "he-IL"
    case thTH = "th-TH"
    case nnNO = "nn-NO"

    // MARK: - Sanity checks

    /// Debug-build assertion that the enum still matches PLAN.md §2's 40-locale
    /// count. Trips early in DEBUG builds if a case is added or removed without
    /// the matching PLAN.md update. The runtime test in `ParityTests.swift`
    /// (Phase F follow-up) is the release-build counterpart.
    #if DEBUG
    @inlinable
    public static func _debugAssertCount() {
        assert(
            Language.allCases.count == 40,
            "Language.allCases.count must be 40 per PLAN.md §2 (got \(Language.allCases.count))"
        )
    }
    #endif
}
