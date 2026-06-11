// PreprocessSession — runs audio → mel on the Core AI ANE/GPU.
//
// The .aimodel's "preprocess_audio" function takes a fixed-shape audio
// buffer of 20480 samples (1.28 s @ 16 kHz, matching the encoder's 128-frame
// chunk) and returns [1, 128, 128] mel features. The matmul/conv1d-based
// preprocessor in convert/nemotron_convert/preproc_matmul.py is bit-exact
// against NeMo's torch.stft path on real audio — no Swift-side mel parity
// concerns.

import CoreAI
import Foundation

public actor PreprocessSession {
    let function: InferenceFunction
    /// Samples per call. Fixed by the converted .aimodel; matches the
    /// encoder's 128 mel frame chunk (128 × hop=160 = 20480).
    public static let audioSamples: Int = 20320

    public init(model: AIModel) async throws {
        guard let fn = try model.loadFunction(named: "preprocess_audio") else {
            throw ASRError.missingFunction("preprocess_audio")
        }
        self.function = fn
    }

    /// Convert 20480 raw audio samples to [1, 128, 128] mel features via the
    /// baked-in preprocessor. Caller is responsible for padding short
    /// buffers to exactly 20480 samples.
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
