// PreprocessSession — runs audio → mel on the Core AI ANE/GPU.
//
// The exported `preprocess_audio` function takes 20832 samples:
// 512 left-context samples + 20320 chunk samples. The left context lets
// non-first chunks use REAL audio from the previous chunk on their left
// edge instead of reflect-padding the chunk's own start (which would
// introduce mel discontinuities at every chunk boundary and drop words
// that fall near them). For chunk 0 the caller passes zeros as left
// context — equivalent to NeMo's `torch.stft(center=True)` at the start
// of an utterance (silence reflects to silence).
//
// The matmul/conv1d-based preprocessor in convert/nemotron_convert/preproc_matmul.py
// is bit-exact against NeMo's `torch.stft` path on real audio — no
// Swift-side mel parity concerns.

import CoreAI
import Foundation

public actor PreprocessSession {
    let function: InferenceFunction

    /// Total input samples per call.
    public static let audioSamples: Int = 20832

    /// Left-context samples at the start of the input (filled with previous
    /// chunk's tail or zeros for chunk 0).
    public static let leftContextSamples: Int = 512

    /// Real-audio samples per chunk (the part the caller advances by between
    /// successive calls).
    public static let chunkSamples: Int = 20320

    public init(model: AIModel) async throws {
        guard let fn = try model.loadFunction(named: "preprocess_audio") else {
            throw ASRError.missingFunction("preprocess_audio")
        }
        self.function = fn
    }

    /// Convert [1, 20832] audio (context + chunk) to [1, 128, 128] mel
    /// features. Caller is responsible for filling the input NDArray with
    /// the appropriate context+chunk samples.
    public func extract(audio: NDArray, length: NDArray) async throws -> NDArray {
        var outputs = try await function.run(
            inputs: ["audio": audio, "audio_length": length],
            states: InferenceFunction.MutableViews(),
        )
        guard let mel = outputs.remove("mel")?.ndArray else {
            throw ASRError.missingOutput("mel")
        }
        return mel
    }
}
