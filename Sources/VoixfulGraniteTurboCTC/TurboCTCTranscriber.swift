// VoixfulGraniteTurboCTC — IBM Granite Speech 5.0 TurboCTC (470M, pure CTC,
// English) as a `Transcribing` backend behind the Voixful façade.
// Full-context: buffers all input via `consume`, transcribes once at
// `finalize`. The lifecycle is the shared `FullContextTranscriber`; this
// target only supplies the engine + its metadata.

import Foundation
import VoixfulEngine
import VoixfulFullContext

extension TurboCTCEngine: FullContextEngine {
    public static let displayName = "Granite Speech 5.0 TurboCTC"
    public static let identifierSuffix = "turboctc"
    public static let supportedLocales = [Locale(identifier: "en-US")]
}

public typealias TurboCTCTranscriber = FullContextTranscriber<TurboCTCEngine>
