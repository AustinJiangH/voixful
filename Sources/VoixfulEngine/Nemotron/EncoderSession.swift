// Phase D — Core AI session actor for the cache-aware FastConformer encoder.
//
// Wraps the "encode" inference function from `nemotron-streaming.aimodel`.
// Shapes verified 2026-06-11 against the converted English-only checkpoint
// (`nvidia/nemotron-speech-streaming-en-0.6b`). The multilingual variant
// (`nvidia/nemotron-3.5-asr-streaming-0.6b`) requires NeMo's unreleased
// `EncDecRNNTBPEModelWithPrompt` and is not exercisable today.
//
// Three streaming caches (NOT four — the published checkpoint has no
// `pre_cache`; the PLAN doc's mention of it was extrapolation from the
// soniqo Core ML port for the multilingual model). Cache layout is
// batch-first (`[B, n_layers, ...]`) to match what `forward_for_export`
// consumes — see convert/nemotron_convert/models.py for the exact tensors.

import CoreAI
import Foundation

public actor EncoderSession {
    /// The "encode" inference function. Loaded once at init; immutable.
    let function: InferenceFunction

    // MARK: Streaming cache state

    /// Per-layer self-attention K/V cache.
    /// Shape `[1, 24, 70, 1024]` (batch, layers, K/V slots, d_model).
    var cacheLastChannel: NDArray

    /// Per-layer convolution left-pad cache.
    /// Shape `[1, 24, 1024, 8]`.
    var cacheLastTime: NDArray

    /// Running attention-cache length. Shape `[1]`, dtype int64.
    var cacheLastChannelLen: NDArray

    // MARK: - Init

    public init(model: AIModel, functionName: String = "encode") async throws {
        guard let fn = try model.loadFunction(named: functionName) else {
            throw ASRError.missingFunction(functionName)
        }
        self.function = fn

        // Shapes verified against the loaded NeMo encoder's
        // `get_initial_cache_state` / `input_example` (batch-first layout).
        self.cacheLastChannel = NDArray(shape: [1, 24, 70, 1024], scalarType: .float32)
        self.cacheLastTime = NDArray(shape: [1, 24, 1024, 8], scalarType: .float32)
        // The converted .aimodel reports this state as int32 even though
        // NeMo's get_initial_cache_state returns int64 — the coreai_torch
        // export lowers it during conversion. Match the .aimodel contract.
        self.cacheLastChannelLen = NDArray(shape: [1], scalarType: .int32)
    }

    // MARK: - Forward

    /// Run one encoder chunk.
    ///
    /// - Parameters:
    ///   - mel: Mel spectrogram chunk, shape `[1, 128, T]` where T is a
    ///     multiple of 8 in {128, 136, ..., 256}. The fixed-shape `.aimodel`
    ///     produced by `convert.py` today targets T=128.
    ///   - melLength: Unpadded frame count as a 1-D int32 NDArray `[1]`.
    /// - Returns: Encoder output, shape `[1, T_out, 1024]` (channels-last;
    ///   the convert wrapper transposes from NeMo's channels-second output).
    public func forward(mel: NDArray, melLength: NDArray) async throws -> NDArray {
        var states = InferenceFunction.MutableViews()
        states.insert(&cacheLastChannel, for: "cache_last_channel")
        states.insert(&cacheLastTime, for: "cache_last_time")
        states.insert(&cacheLastChannelLen, for: "cache_last_channel_len")

        var outputs = try await function.run(
            inputs: ["mel_chunk": mel, "mel_length": melLength],
            states: states
        )

        guard let encOut = outputs.remove("enc_out")?.ndArray else {
            throw ASRError.missingOutput("enc_out")
        }
        return encOut
    }

    // MARK: - Reset

    /// Zero all three streaming caches at utterance boundary.
    /// PLAN.md §11 Anti-pattern 11: reset all together or none.
    public func reset() async {
        cacheLastChannel.zeroFloat32()
        cacheLastTime.zeroFloat32()
        cacheLastChannelLen.zeroInt32()
    }
}
