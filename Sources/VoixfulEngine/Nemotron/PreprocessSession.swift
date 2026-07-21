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

    /// Left-context samples at the start of the input (filled with previous
    /// chunk's tail or zeros for chunk 0).
    public static let leftContextSamples: Int = 512

    /// Mel frames this variant emits per call (128 canonical; 256 pairs with
    /// `encode_256` so the batch path halves its encoder invocations).
    public let frames: Int
    /// Real-audio samples per chunk = frames * 160 (10 ms hop).
    public let chunkSamples: Int
    /// Total input samples per call = left context + chunk.
    public let audioSamples: Int

    public init(model: AIModel, frames: Int = 128) async throws {
        let name = frames == 128 ? "preprocess_audio" : "preprocess_audio_\(frames)"
        guard let fn = try model.loadFunction(named: name) else {
            throw ASRError.missingFunction(name)
        }
        self.function = fn
        self.frames = frames
        self.chunkSamples = frames * 160
        self.audioSamples = Self.leftContextSamples + frames * 160
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
