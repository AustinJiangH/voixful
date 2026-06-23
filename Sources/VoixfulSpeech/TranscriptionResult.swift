// VoixfulSpeech — result + option types.
//
// These mirror Apple's `SpeechTranscriber.Result` and its option sets, with our
// own names. Model-agnostic; shared by every backend.

import Foundation
import CoreMedia

/// A transcription result emitted on a transcriber's `results` stream.
///
/// `@unchecked Sendable`: every stored property is a value type or `Sendable`
/// reference; `CMTimeRange` is a pure value type whose `Sendable`-ness is not
/// always declared by CoreMedia, so we vouch for it explicitly.
public struct TranscriptionResult: @unchecked Sendable {
    /// The transcription text. When `.audioTimeRange` is requested, runs carry
    /// the `\.voixful.audioTimeRange` attribute (see `AudioTimeRange`).
    public let text: AttributedString

    /// Audio time span this result covers, on the session timeline.
    public let range: CMTimeRange

    /// `true` once committed (will not be revised); `false` for a volatile
    /// result that a later result may replace. Mirrors Apple's volatile/final.
    public let isFinal: Bool

    /// Alternative transcriptions, when `.alternativeTranscriptions` is set.
    public let alternatives: [TranscriptionAlternative]

    public init(
        text: AttributedString,
        range: CMTimeRange,
        isFinal: Bool,
        alternatives: [TranscriptionAlternative] = []
    ) {
        self.text = text
        self.range = range
        self.isFinal = isFinal
        self.alternatives = alternatives
    }
}

/// An alternative transcription hypothesis.
public struct TranscriptionAlternative: Sendable {
    public let text: AttributedString
    public let confidence: Double?
    public init(text: AttributedString, confidence: Double? = nil) {
        self.text = text
        self.confidence = confidence
    }
}

// MARK: - Options

/// How results are reported. Mirrors `SpeechTranscriber.ReportingOption`.
public struct ReportingOption: OptionSet, Sendable, Hashable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }

    /// Emit volatile (in-progress, revisable) results in addition to finals.
    public static let volatileResults = ReportingOption(rawValue: 1 << 0)
    /// Include alternative hypotheses on results.
    public static let alternativeTranscriptions = ReportingOption(rawValue: 1 << 1)
    /// Prefer lower latency over maximal context.
    public static let fastResults = ReportingOption(rawValue: 1 << 2)
}

/// Per-run attributes attached to result text. Mirrors
/// `SpeechTranscriber.ResultAttributeOption`.
public struct ResultAttributeOption: OptionSet, Sendable, Hashable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }

    /// Attach `\.voixful.audioTimeRange` to result-text runs.
    public static let audioTimeRange = ResultAttributeOption(rawValue: 1 << 0)
    /// Attach confidence scores (when the backend provides them).
    public static let transcriptionConfidence = ResultAttributeOption(rawValue: 1 << 1)
}

// MARK: - audioTimeRange attribute

/// Audio time span of an `AttributedString` run, in seconds on the session
/// timeline. (Stored as `Double` seconds rather than `CMTimeRange` so the value
/// satisfies `AttributedStringKey`'s `Hashable` requirement without depending on
/// CoreMedia conformances; use `cmTimeRange` for the CoreMedia form.)
public struct AudioTimeRange: Hashable, Sendable, Codable {
    public var start: Double
    public var duration: Double
    public init(start: Double, duration: Double) {
        self.start = start
        self.duration = duration
    }
    public var cmTimeRange: CMTimeRange {
        CMTimeRange(
            start: CMTime(seconds: start, preferredTimescale: 1000),
            duration: CMTime(seconds: duration, preferredTimescale: 1000)
        )
    }
}

extension AttributeScopes {
    /// Voixful's custom attribute scope. Access via `\.voixful.audioTimeRange`.
    public struct VoixfulAttributes: AttributeScope {
        public let audioTimeRange: AudioTimeRangeAttribute

        public enum AudioTimeRangeAttribute: AttributedStringKey {
            public typealias Value = AudioTimeRange
            public static let name = "com.voixful.audioTimeRange"
        }
    }

    public var voixful: VoixfulAttributes.Type { VoixfulAttributes.self }
}

extension AttributeDynamicLookup {
    public subscript<T: AttributedStringKey>(
        dynamicMember keyPath: KeyPath<AttributeScopes.VoixfulAttributes, T>
    ) -> T {
        self[T.self]
    }
}
