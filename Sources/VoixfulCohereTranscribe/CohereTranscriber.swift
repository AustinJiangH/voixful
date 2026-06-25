// VoixfulCohereTranscribe — Cohere Transcribe 2B (attention encoder-decoder) as a
// `Transcribing` backend behind the Voixful façade. Full-context: buffers all
// input via `consume`, transcribes once at `finalize`. The lifecycle is the shared
// `FullContextTranscriber`; this target only supplies the engine + its metadata.

import Foundation
import VoixfulEngine
import VoixfulFullContext

extension CohereTranscribeEngine: FullContextEngine {
    public static let displayName = "Cohere Transcribe 2B"
    public static let identifierSuffix = "cohere"
    // Cohere Transcribe handles 14 languages; the locale is a display/metadata
    // hint — the model is prompted/auto-detects. en-US is the default.
    public static let supportedLocales = [Locale(identifier: "en-US")]
}

public typealias CohereTranscriber = FullContextTranscriber<CohereTranscribeEngine>
