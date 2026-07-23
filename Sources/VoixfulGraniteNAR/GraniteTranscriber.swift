// VoixfulGraniteNAR — IBM Granite Speech 4.1 2B NAR (single-pass CTC + LLM
// transcript editor) as a `Transcribing` backend behind the Voixful façade.
// Full-context: buffers all input via `consume`, transcribes once at
// `finalize`. The lifecycle is the shared `FullContextTranscriber`; this
// target only supplies the engine + its metadata.

import Foundation
import VoixfulEngine
import VoixfulFullContext

extension GraniteNAREngine: FullContextEngine {
    public static let displayName = "Granite Speech 4.1 2B NAR"
    public static let identifierSuffix = "granite"
    // EN/FR/DE/ES/PT per the model card; locale is a display/metadata hint.
    public static let supportedLocales = [Locale(identifier: "en-US")]
}

public typealias GraniteTranscriber = FullContextTranscriber<GraniteNAREngine>
