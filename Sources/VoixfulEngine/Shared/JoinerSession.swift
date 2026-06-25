// Phase D — Core AI session actor for the RNN-T joiner.
//
// Wraps the "join" inference function from `nemotron-streaming.aimodel`.
// Stateless: the joiner is a pure function of (encoder_step, prediction_embedding)
// → vocab logits. No `register_buffer` on the PyTorch side, no `state_names` in
// the convert call, no `MutableViews` here. The actor exists purely to serialize
// access to the underlying `InferenceFunction` and to live alongside its sibling
// sessions in a single `NemotronStreamingEngine` actor graph.
//
// Input names ("enc_step", "pred_emb") match `input_names=[...]` for the
// "join" program in convert/nemotron_convert/convert.py.

import CoreAI
import Foundation

public actor JoinerSession {
    /// The "join" inference function. Loaded once at init; immutable.
    let function: InferenceFunction

    // MARK: - Init

    public init(model: AIModel) async throws {
        // VERIFY: Core AI API surface — confirm against Xcode 27 SDK.
        guard let fn = try model.loadFunction(named: "join") else {
            throw ASRError.missingFunction("join")
        }
        self.function = fn
    }

    // MARK: - Forward

    /// Combine one encoder time-step with one prediction embedding to produce
    /// vocab logits.
    ///
    /// - Parameters:
    ///   - enc: A single encoder time-step slice, shape `[1, 1, d_model]`.
    ///     Caller indexes the full encoder output along the time axis before
    ///     calling — the joiner sees one frame at a time per the RNN-T greedy
    ///     loop in PLAN.md §6.
    ///   - pred: Prediction network embedding from `PredictionSession.step(...)`,
    ///     shape `[1, 1, pred_hidden=640]`.
    /// - Returns: Vocab logits, shape `[1, 1, 1, vocab=13087]`.
    public func forward(enc: NDArray, pred: NDArray) async throws -> NDArray {
        // No states for the joiner. Pass empty MutableViews to match the
        // `function.run(inputs:states:)` signature uniformly with Encoder /
        // Prediction sessions.
        //
        // VERIFY: Core AI API surface — does `function.run` allow omitting
        // `states:` entirely when there are none, or must we pass an empty
        // `MutableViews()`? PLAN.md §4's snippet always passes one; keep
        // symmetric for now.
        var outputs = try await function.run(
            inputs: ["enc_step": enc, "pred_emb": pred],
            states: InferenceFunction.MutableViews()
        )

        guard let logits = outputs.remove("logits")?.ndArray else {
            throw ASRError.missingOutput("logits")
        }
        return logits
    }

    // MARK: - Argmax helper

    /// Find the vocabulary index with the maximum logit value.
    ///
    /// Used by the greedy RNN-T loop in PLAN.md §6 to pick the next emitted
    /// token. Returns Int32 to match the token-id type expected by
    /// `PredictionSession.step(token:)`.
    ///
    /// Free function on the actor (not an instance method needing state) — but
    /// kept on the actor for locality with `forward(...)`. Nonisolated so the
    /// greedy loop can call it without an extra `await` per token.
    ///
    /// VERIFY: NDArray scalar accessor API (e.g., `.scalars()`,
    /// `.view().withUnsafeBufferPointer`). The snippet below uses a plausible
    /// `view().withUnsafeBufferPointer(of:)` shape consistent with the
    /// `mutableView()` accessors used in EncoderSession/PredictionSession;
    /// replace with the documented accessor once Xcode 27 SDK is on disk.
    public nonisolated func argmax(_ logits: NDArray) -> Int32 {
        // Resolves the Phase-D VERIFY: SDK 27.0 exposes read access via
        // `view(as:).withUnsafePointer { ptr, shape, strides in ... }`. The
        // `argmaxFloat32()` helper in NDArrayHelpers.swift wraps the closure +
        // count math so this site stays a one-liner. The input is expected to
        // be `[1, 1, 1, vocab]` contiguous Float32 from the join function.
        return logits.argmaxFloat32()
    }
}
