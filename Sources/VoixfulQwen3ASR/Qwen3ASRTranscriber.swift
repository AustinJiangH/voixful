// VoixfulQwen3ASR — Alibaba/Qwen **Qwen3-ASR** (SALM: audio encoder + Qwen3 LLM
// decoder) as a `Transcribing` backend behind the Voixful façade. Full-context;
// lifecycle via the shared `FullContextTranscriber`. Two checkpoints share this
// one backend/runtime — qwen3-asr-0.6b and qwen3-asr-1.7b — differing only in the
// converted asset (like Parakeet v2/v3). Multilingual (52 languages/dialects);
// pure ASR (no Canary-style second formatting pass).

import Foundation
import VoixfulEngine
import VoixfulFullContext

extension Qwen3ASREngine: FullContextEngine {
    public static let displayName = "Qwen3-ASR"
    public static let identifierSuffix = "qwen3asr"
    // Qwen3-ASR is multilingual (52 langs); en-US is the primary locale we drive
    // and the only one benchmarked so far. Others are reachable once verified.
    public static let supportedLocales = [Locale(identifier: "en-US")]
}

public typealias Qwen3ASRTranscriber = FullContextTranscriber<Qwen3ASREngine>
