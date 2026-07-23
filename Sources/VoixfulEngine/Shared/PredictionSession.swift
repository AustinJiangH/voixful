// Phase D / Phase E — Core AI session actor for the RNN-T prediction network.
//
// Wraps the "predict_step" inference function from `nemotron-streaming.aimodel`.
// The prediction network is a 2-layer LSTM with `pred_hidden = 640`.
// Its hidden and cell states are the only state this session owns; they are
// allocated once and updated in place by Core AI via `MutableViews` on every
// `step(...)`.
//
// State names ("lstm_h", "lstm_c") match the `state_names=[...]` list passed to
// `converter.add_exported_program(...)` for the `predict_step` program in
// convert/nemotron_convert/convert.py.
//
// Phase E amendment: GreedyRNNTDecoder needs the *current* prediction embedding
// to feed the joiner BEFORE it knows whether the joiner will emit a non-blank
// (and thus call `step(...)` to advance the LSTM) or a blank (no advance). RNN-T
// decoding therefore needs a "last embedding" cached on the prediction session
// that the joiner can read at every joiner step, and that only mutates when a
// non-blank emit happens. We add `lastEmbedding` as actor-scoped state with the
// contract:
//   - At `reset()` / at session start, `lastEmbedding` is the zero vector of
//     shape `[1, 1, 640]` — RNN-T convention for "no previous emitted token,
//     start of utterance".
//   - After every successful `step(token:)`, `lastEmbedding` holds the new
//     embedding produced by feeding `token` through the LSTM.
//   - `currentEmbedding()` is the read accessor the decoder calls.

import CoreAI
import Foundation

public actor PredictionSession {
    /// The "predict_step" inference function. Loaded once at init; immutable.
    let function: InferenceFunction

    // MARK: LSTM state
    //
    // 2-layer LSTM, hidden size 640. Shape convention `[num_layers, batch, hidden]`
    // matches PyTorch's `torch.nn.LSTM` default, which is what the NeMo prediction
    // network uses under the hood and what the Phase B PyTorch wrapper exposes via
    // `register_buffer("lstm_h", ...)` / `register_buffer("lstm_c", ...)`.

    /// LSTM hidden state. Shape `[2, 1, 640]`.
    var lstmH: NDArray

    /// LSTM cell state. Shape `[2, 1, 640]`.
    var lstmC: NDArray

    // MARK: Prediction embedding cache (Phase E)
    //
    // The last embedding produced by `step(token:)` — or a zero buffer at start
    // of utterance / after `reset()`. The greedy RNN-T decoder reads this on
    // every joiner step (inner loop) without advancing the LSTM,
    // because the joiner sees one (enc_step, pred_emb) pair per try and only
    // a non-blank emit causes us to call `step(...)` and overwrite this.
    //
    // VERIFY: pred embedding shape. `pred_hidden = 640` and the
    // PyTorch wrapper in convert/nemotron_convert/models.py is the source of
    // truth for whether the output is `[1, 640]`, `[1, 1, 640]`, or something
    // else. The joiner doc comment in JoinerSession.swift says `[1, 1, 640]`,
    // so we use that. If models.py disagrees, change BOTH this shape AND the
    // JoinerSession doc comment to match.

    /// The most recent prediction embedding. Shape `[1, 1, 640]`.
    /// Zero-initialized at session start and after `reset()`.
    ///
    /// Exposed as `public private(set)` so the decoder can read it directly
    /// without an extra accessor call (it still goes through actor isolation;
    /// see `currentEmbedding()` for the documented read API).
    public private(set) var lastEmbedding: NDArray

    // MARK: Per-call scratch buffer
    //
    // The decoder calls `step(token:)` once per emitted non-blank token. Per
    // the anti-pattern "don't allocate a fresh NDArray each chunk",
    // hoist the 1-element prev_token buffer to actor-scoped storage so we
    // overwrite a single allocation rather than churning the allocator on every
    // greedy step.

    /// Scratch buffer for the `prev_token` input. Shape `[1]` int32.
    /// Overwritten with the new token id at the top of every `step(token:)`.
    var prevTokenScratch: NDArray

    // MARK: - Init

    public init(model: AIModel) async throws {
        // VERIFY: Core AI API surface — confirm against Xcode 27 SDK.
        guard let fn = try model.loadFunction(named: "predict_step") else {
            throw ASRError.missingFunction("predict_step")
        }
        self.function = fn

        // Allocate LSTM state buffers exactly once. Reused across every
        // `step(...)` via MutableViews.
        //
        // VERIFY: Core AI API surface — NDArray(shape:scalarType:) initializer
        // and .float32 scalar-type spelling.
        self.lstmH = NDArray(shape: [2, 1, 640], scalarType: .float32)
        self.lstmC = NDArray(shape: [2, 1, 640], scalarType: .float32)

        // Zero-init prediction embedding. RNN-T convention: at utterance start,
        // before any non-blank has been emitted, the prediction network output
        // is the all-zeros vector. The greedy decoder feeds this through the
        // joiner at t=0 to get the first set of logits.
        self.lastEmbedding = NDArray(shape: [1, 1, 640], scalarType: .float32)

        // Pre-allocate the 1-element prev_token scratch buffer.
        // [1, 1] matches the exported predict_step's input contract: NeMo's
        // RNNTDecoder.predict expects targets shaped [B, U].
        self.prevTokenScratch = NDArray(shape: [1, 1], scalarType: .int32)
    }

    // MARK: - Read accessor

    /// The most recent prediction embedding, suitable for feeding into
    /// `JoinerSession.forward(enc:pred:)`.
    ///
    /// Equivalent to reading `lastEmbedding` directly; provided as an explicit
    /// method so the greedy decode call site reads as "ask prediction for its
    /// current embedding" rather than "reach into the prediction's internal
    /// state". Both compile to the same actor hop.
    public func currentEmbedding() -> NDArray {
        return lastEmbedding
    }

    // MARK: - Step

    /// Advance the prediction network by one token. Updates the LSTM hidden +
    /// cell states in place AND overwrites `lastEmbedding` with the new
    /// embedding so the next joiner step sees the post-emit pred output.
    ///
    /// Called inside the RNN-T greedy decode loop every time the
    /// joiner emits a non-blank symbol.
    ///
    /// - Parameter token: The previously emitted non-blank token id.
    public func step(token: Int32) async throws {
        // Resolves the Phase-D VERIFY on the single-element write path. The
        // `setInt32(at:to:)` helper in NDArrayHelpers.swift wraps
        // `mutableView(as: Int32.self).withUnsafeMutablePointer { ... }` so the
        // call site stays a one-liner.
        prevTokenScratch.setInt32(at: 0, to: token)

        // VERIFY: Core AI API surface — MutableViews state-name keys must match
        // the `state_names=[...]` list for the `predict_step` program in
        // convert/nemotron_convert/convert.py.
        var states = InferenceFunction.MutableViews()
        states.insert(&lstmH, for: "lstm_h")
        states.insert(&lstmC, for: "lstm_c")

        var outputs = try await function.run(
            inputs: ["prev_token": prevTokenScratch],
            states: states
        )

        guard let predEmb = outputs.remove("pred_emb")?.ndArray else {
            throw ASRError.missingOutput("pred_emb")
        }

        // Cache the new embedding so the next joiner call sees the post-emit
        // pred output. Reassignment here is correct: `lastEmbedding` is NOT
        // registered as a `MutableViews` state on the predict_step function —
        // it is the function's OUTPUT, not a state buffer. Anti-pattern 4
        // applies to state buffers (whose registration would be invalidated by
        // reassignment); the embedding output has no such constraint.
        self.lastEmbedding = predEmb
    }

    // MARK: - Reset

    /// Zero the LSTM hidden and cell states AND the cached embedding at
    /// utterance boundary.
    ///
    /// Per the "Utterance boundary" anti-pattern: must be called
    /// at the same instant the encoder caches are reset, never independently.
    public func reset() async {
        // Resolves the Phase-D VERIFY: SDK 27.0 uses
        // `mutableView(as:).withUnsafeMutablePointer { ... }`. The `zeroFloat32`
        // helper in NDArrayHelpers.swift centralizes the allocation-free fill.
        lstmH.zeroFloat32()
        lstmC.zeroFloat32()

        // Reset the cached embedding back to the start-of-utterance zero
        // vector. Reallocating here (rather than `fill(0)`-ing the existing
        // buffer) is safe because `lastEmbedding` is not registered with any
        // MutableViews binding — `step(...)` reassigns it from each call's
        // `pred_emb` output, so its identity is already non-stable. Using
        // a fresh allocation guarantees clean shape even if a prior `step(...)`
        // returned an unexpected shape.
        self.lastEmbedding = NDArray(shape: [1, 1, 640], scalarType: .float32)
    }
}
