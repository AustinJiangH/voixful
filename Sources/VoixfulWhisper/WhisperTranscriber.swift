// VoixfulWhisper — OpenAI Whisper (attention encoder-decoder) as a `Transcribing`
// backend behind the Voixful façade. Full-context: buffers all input via
// `consume`, transcribes once at `finalize`. The lifecycle is the shared
// `FullContextTranscriber`; this target only supplies the engine + its metadata.

import Foundation
import VoixfulEngine
import VoixfulFullContext

extension WhisperEngine: FullContextEngine {
    public static let displayName = "Whisper"
    public static let identifierSuffix = "whisper"
    // Whisper is multilingual; the locale is a display/metadata hint — the model
    // auto-detects (or is primed via the language token). en-US is the default.
    public static let supportedLocales = [Locale(identifier: "en-US")]
}

public typealias WhisperTranscriber = FullContextTranscriber<WhisperEngine>
