// VoixfulParakeetTDT — Parakeet TDT 0.6B as a `Transcribing` backend behind the
// Voixful façade. Full-context: buffers all input via `consume`, transcribes once
// at `finalize`. The lifecycle is the shared `FullContextTranscriber`; this target
// only supplies the engine + its display metadata.

import Foundation
import VoixfulEngine
import VoixfulFullContext

extension ParakeetTDTEngine: FullContextEngine {
    public static let displayName = "Parakeet TDT 0.6B"
    public static let identifierSuffix = "parakeet"
    // Parakeet TDT v3 is multilingual; the locale is a display/metadata hint —
    // the model auto-detects language. en-US is the default.
    public static let supportedLocales = [Locale(identifier: "en-US")]
}

public typealias ParakeetTDTTranscriber = FullContextTranscriber<ParakeetTDTEngine>
