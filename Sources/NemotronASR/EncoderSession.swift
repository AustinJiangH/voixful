// Phase D — Core AI session actor for the cache-aware FastConformer encoder.
//
// Wraps the "encode" inference function from `nemotron-streaming.aimodel`
// (produced by the Phase B convert pipeline). Owns the four streaming cache
// tensors as `NDArray` state buffers; they are allocated exactly once at init
// and updated in place by Core AI via `MutableViews` on every `forward(...)`.
//
// Shapes and dtypes come from PLAN.md §2 ("Streaming caches"). The exact
// state names ("pre_cache", "cache_last_channel", "cache_last_time",
// "cache_last_channel_len") match the `state_names=[...]` list passed to
// `converter.add_exported_program(...)` in convert/nemotron_convert/convert.py.
//
// Per PLAN.md §4 / Anti-pattern 4: NEVER reassign these buffers. `reset()`
// zeros them in place via `mutableView().fill(0)` so the underlying allocation
// (and the `MutableViews` registration) remains stable across utterances.

import CoreAI
import Foundation

public actor EncoderSession {
    /// The "encode" inference function. Loaded once at init; immutable.
    let function: InferenceFunction

    // MARK: Streaming cache state
    //
    // These are the four tensors the cache-aware FastConformer round-trips
    // across chunks. Lifetimes:
    //   - Allocated in `init` with the canonical shapes from PLAN.md §2.
    //   - Registered with `InferenceFunction.MutableViews` on every `forward`.
    //   - Zeroed in place by `reset()` at utterance boundary.
    //   - Never reassigned; never reallocated.

    /// Trailing mel frames carried into the next chunk. Shape `[1, 128, 9]`.
    var preCache: NDArray

    /// Per-layer self-attention K/V cache. Shape `[24, 1, 56, 1024]`.
    ///
    /// VERIFY: d_model 1024 vs 512 per PLAN.md §10 — HF model card says 512,
    /// soniqo/speech-swift's working Core ML port used 1024. PLAN.md hypothesizes
    /// 512 is the projection dim and 1024 is the encoder hidden; Phase 1 confirms
    /// by inspecting the loaded `.nemo` config. If the checkpoint actually wants
    /// 512 here, change the last axis to 512 AND update the matching
    /// `register_buffer` shape in `convert/nemotron_convert/wrappers.py`.
    var cacheLastChannel: NDArray

    /// Per-layer convolution left-pad cache. Shape `[24, 1, 1024, 8]`.
    /// Same d_model caveat as `cacheLastChannel` — the 1024 here must match.
    var cacheLastTime: NDArray

    /// Running attention-cache length (int32 scalar wrapped in a 1-D array).
    /// Shape `[1]`. Incremented in place by the encoder graph each call.
    var cacheLastChannelLen: NDArray

    // MARK: - Init

    public init(model: AIModel) async throws {
        // VERIFY: Core AI API surface — confirm against Xcode 27 SDK.
        // PLAN.md §4 shows `try model.loadFunction(named: "encode")!`; we treat
        // a nil return as a missing function and convert to a typed error.
        guard let fn = try model.loadFunction(named: "encode") else {
            throw ASRError.missingFunction("encode")
        }
        self.function = fn

        // Allocate the four streaming caches exactly once. NDArrays are reused
        // across chunks via MutableViews — see Anti-pattern 4 in PLAN.md §11.
        //
        // VERIFY: Core AI API surface — confirm `NDArray(shape:scalarType:)`
        // initializer and `.float32` / `.int32` scalar-type spellings against
        // Xcode 27 SDK. Snippet form taken from PLAN.md §4.
        self.preCache = NDArray(shape: [1, 128, 9], scalarType: .float32)
        self.cacheLastChannel = NDArray(shape: [24, 1, 56, 1024], scalarType: .float32)
        self.cacheLastTime = NDArray(shape: [24, 1, 1024, 8], scalarType: .float32)
        self.cacheLastChannelLen = NDArray(shape: [1], scalarType: .int32)
    }

    // MARK: - Forward

    /// Run one encoder chunk. Updates the four streaming caches in place via
    /// `MutableViews` and returns the encoder output for this chunk.
    ///
    /// - Parameters:
    ///   - mel: Mel spectrogram chunk, shape `[1, 128, T]` where T is the
    ///     chunk's frame count (8 / 16 / 32 / 56 / 112 per PLAN.md §2 chunk-size dial).
    ///   - langId: 128-dim one-hot language ID, shape `[1, 128]`.
    /// - Returns: Encoder output for this chunk (`enc_out`). Caller is
    ///   responsible for indexing across the time axis in the RNN-T loop.
    public func forward(mel: NDArray, langId: NDArray) async throws -> NDArray {
        // VERIFY: Core AI API surface — `InferenceFunction.MutableViews()`
        // default-init + `.insert(_:for:)` per PLAN.md §4 snippet. The state
        // names here MUST match the `state_names=[...]` list in
        // convert/nemotron_convert/convert.py for the "encode" program.
        var states = InferenceFunction.MutableViews()
        states.insert(&preCache, for: "pre_cache")
        states.insert(&cacheLastChannel, for: "cache_last_channel")
        states.insert(&cacheLastTime, for: "cache_last_time")
        states.insert(&cacheLastChannelLen, for: "cache_last_channel_len")

        // VERIFY: Core AI API surface — `function.run(inputs:states:)` return
        // type and `outputs.remove(_:)?.ndArray` accessor per PLAN.md §4.
        var outputs = try await function.run(
            inputs: ["mel_chunk": mel, "lang_id": langId],
            states: states
        )

        guard let encOut = outputs.remove("enc_out")?.ndArray else {
            throw ASRError.missingOutput("enc_out")
        }
        return encOut
    }

    // MARK: - Reset

    /// Zero all four streaming caches at utterance boundary.
    ///
    /// Per PLAN.md §4 ("Utterance boundary") and Anti-pattern 11: all four
    /// caches reset together, or none of them. Callers must invoke
    /// `PredictionSession.reset()` alongside this on the same boundary.
    ///
    /// Uses `mutableView().fill(0)` so the underlying allocation and the
    /// MutableViews registration stay stable — never reassign these properties.
    public func reset() async {
        // Resolves the Phase-D VERIFY: SDK 27.0 exposes element access via
        // `mutableView(as:).withUnsafeMutablePointer { ptr, shape, strides in ... }`.
        // The `zeroFloat32` / `zeroInt32` helpers in NDArrayHelpers.swift centralize
        // the (allocation-free) fill so this site stays a single-line semantic.
        preCache.zeroFloat32()
        cacheLastChannel.zeroFloat32()
        cacheLastTime.zeroFloat32()
        cacheLastChannelLen.zeroInt32()
    }
}
