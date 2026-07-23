// VoixfulCanaryQwen — NVIDIA Canary-Qwen 2.5B (SALM: Conformer + Qwen3-1.7B)
// as a `Transcribing` backend behind the Voixful façade. Full-context;
// lifecycle via the shared `FullContextTranscriber`. Uniquely, this backend
// can FORMAT its transcript (markdown bullets) via a second single-weight
// LLM pass — enable with VOIXFUL_CANARY_FORMAT=bullets or the CLI --bullets.

import Foundation
import VoixfulEngine
import VoixfulFullContext

extension CanaryQwenEngine: FullContextEngine {
    public static let displayName = "Canary-Qwen 2.5B"
    public static let identifierSuffix = "canary"
    public static let supportedLocales = [Locale(identifier: "en-US")]
}

public typealias CanaryTranscriber = FullContextTranscriber<CanaryQwenEngine>
