// Phase E — Greedy RNN-T decoder.
//
// Implements the encoder-time / prediction-step loop from PLAN.md §6:
//
//     for t in 0..<encoderOut.timeSteps {
//         var symbolsThisStep = 0
//         while symbolsThisStep < maxSymbolsPerStep {
//             let logits = try await joiner.forward(enc: encOut[t], pred: prediction.lastEmbedding)
//             let token = joiner.argmax(logits)
//             if token == blankId { break }                  // advance encoder time
//             emit(token)
//             try await prediction.step(token: token)        // advance LSTM state
//             symbolsThisStep += 1
//         }
//     }
//
// The decoder owns:
//   - References to a `PredictionSession` and `JoinerSession`. NOT the encoder —
//     encoder outputs are passed in (the decoder is encoder-step-agnostic so the
//     same decoder instance can be driven by either a streaming chunked encoder
//     pipeline or a one-shot non-streaming encoder pass).
//   - An accumulator of all tokens emitted across this utterance. The decoder
//     instance lives for the lifetime of one utterance; `reset()` clears the
//     accumulator and also resets the prediction LSTM state.
//
// Strict constraints (PLAN.md §11 Anti-patterns):
//   - `maxSymbolsPerStep` is EXACTLY 10 — NeMo's default. Higher risks
//     catastrophic stutter on noise; lower truncates legitimate consonant
//     clusters. NOT a knob.
//   - `blankId` is 1024 for the English-only checkpoint (vocab 0..=1023 are SP
//     tokens, 1024 is blank; total 1025). The multilingual variant uses 13087
//     and a 13088-entry vocab, but it's not exercisable today. NOT a runtime knob.
//   - No beam search. Greedy RNN-T is what Nemotron was trained for; beam
//     doesn't improve transducer quality the way it does CTC/attention.
//
// Phase D coupling: this file required amending `PredictionSession` to add
// `lastEmbedding` / `currentEmbedding()`. See the file header in
// PredictionSession.swift for the contract. No other Phase D files were touched.

import CoreAI
import Foundation

public actor GreedyRNNTDecoder {
    // MARK: - Constants

    /// RNN-T blank token id. Hardcoded to the published checkpoint's value;
    /// NOT a knob (PLAN.md §11 Anti-pattern).
    public static let blankId: Int32 = 1024

    /// Maximum non-blank emits per encoder time step. NeMo's default; the
    /// "stutter safety bound" — without it a misbehaving model can emit the
    /// same non-blank forever and never advance encoder time. EXACTLY 10
    /// (PLAN.md §11 Anti-pattern: not 8, not 12).
    public static let maxSymbolsPerStep: Int = 10

    // MARK: - Dependencies

    /// Prediction network session. The decoder calls
    /// `prediction.currentEmbedding()` once per joiner step (whether the joiner
    /// emits a blank or a non-blank) and `prediction.step(token:)` once per
    /// non-blank emit. Both calls are `await`s into the prediction actor.
    let prediction: PredictionSession

    /// Joiner session. The decoder calls `joiner.forward(enc:pred:)` once per
    /// joiner step and `joiner.argmax(logits)` to pick the next token.
    /// `argmax` is `nonisolated` on JoinerSession so the inner loop only takes
    /// one `await` per joiner step (the `forward` call), not two.
    let joiner: JoinerSession

    // MARK: - Utterance accumulator

    /// All non-blank tokens emitted across this utterance, in emit order.
    /// Cleared by `reset()` on utterance boundary.
    var emittedTokens: [Int32]

    // MARK: - Init

    /// Construct a decoder that drives the given prediction and joiner sessions.
    ///
    /// The encoder is NOT held here: encoder outputs are passed into
    /// `decode(encoderOutput:)`. This lets the same decoder instance be driven
    /// by either a streaming chunked encoder loop (one `decode` call per chunk,
    /// no reset between chunks within the same utterance) or a one-shot
    /// non-streaming pass.
    /// Optional CPU predict/join decoder (Accelerate). When present, the
    /// whole greedy loop runs host-side — zero Core AI dispatches per step
    /// (the ~0.4 ms/dispatch floor measured on this branch). Loaded from the
    /// `<asset>.transducer.json/.bin` sidecar; its manifest carries joint.enc
    /// so raw encoder steps are projected on CPU too.
    let cpu: CPUTransducerDecoder?

    public init(prediction: PredictionSession, joiner: JoinerSession) {
        self.prediction = prediction
        self.joiner = joiner
        self.cpu = nil
        self.emittedTokens = []
    }

    /// Module-internal: `CPUTransducerDecoder` is not public API.
    init(prediction: PredictionSession, joiner: JoinerSession, cpu: CPUTransducerDecoder?) {
        self.prediction = prediction
        self.joiner = joiner
        self.cpu = cpu
        self.emittedTokens = []
    }

    // MARK: - Decode

    /// Drive the RNN-T greedy loop over the time axis of `encoderOutput` and
    /// return the tokens emitted by THIS call (not the accumulator-wide list).
    ///
    /// Callers that drive the decoder chunk-by-chunk concatenate these
    /// per-chunk returns to reconstruct the utterance-wide token sequence. The
    /// utterance-wide sequence is also available as `emittedTokens` for
    /// callers that prefer to read it once at end-of-utterance.
    ///
    /// - Parameter encoderOutput: Encoder output for one or more time steps.
    ///   Expected shape `[1, T, d_model]` (batch, time, feature). The decoder
    ///   iterates t over the time axis.
    /// - Returns: New tokens emitted during this call, in emit order. Empty
    ///   array if the joiner only emitted blanks for every t in this chunk.
    public func decode(encoderOutput: NDArray) async throws -> [Int32] {
        // Snapshot the accumulator length BEFORE we start emitting so we can
        // slice out just this call's contribution at the end. Keeping the
        // accumulator-wide state across calls is the streaming-mode invariant
        // (don't reset between chunks within the same utterance).
        let startIndex = emittedTokens.count

        // Resolves the Phase-E VERIFY: NDArray exposes `shape: [Int]` directly.
        // Encoder output is `[1, T, d_model]`, so the time axis is `shape[1]`.
        let timeSteps = encoderOutput.shape[1]
        let dModel = encoderOutput.shape[2]

        // CPU fast path: no NDArray scratch, no per-step Core AI dispatches.
        if let cpu {
            var newTokens: [Int32] = []
            encoderOutput.view(as: Float.self).withUnsafePointer { src, _, strides in
                for t in 0..<timeSteps {
                    var symbolsThisStep = 0
                    while symbolsThisStep < Self.maxSymbolsPerStep {
                        let (token, _) = cpu.join(encRaw: src.advanced(by: t * strides[1]))
                        if token == Self.blankId { break }
                        newTokens.append(token)
                        cpu.step(token: token)
                        symbolsThisStep += 1
                    }
                }
            }
            emittedTokens.append(contentsOf: newTokens)
            return newTokens
        }

        // Resolves the Phase-E VERIFY on slicing. The SDK 27.0 API has no
        // zero-copy "give me a sub-NDArray pointing at time t" path that's
        // usable with the joiner's `enc: NDArray` signature — slicing lives on
        // View / MutableView (`~Copyable`/`~Escapable`) and the joiner takes a
        // plain NDArray. So we allocate one stable `encStep` scratch NDArray
        // for the chunk and memcpy each time step's `d_model` floats into it
        // via raw pointer copy. One allocation per chunk (cheap), per-t copy
        // is a single `update(from:count:)` of d_model floats (also cheap
        // relative to the inner joiner work).
        var encStep = NDArray(shape: [1, 1, dModel], scalarType: .float32)

        for t in 0..<timeSteps {
            // Copy encoder slice [t] into the encStep scratch. The view's
            // `strides[1]` is the per-time-step element stride; multiplying
            // by `t` gives the offset of time-step t's first element.
            encoderOutput.view(as: Float.self).withUnsafePointer { srcPtr, _, srcStrides in
                let srcOffset = t * srcStrides[1]
                var dstView = encStep.mutableView(as: Float.self)
                dstView.withUnsafeMutablePointer { dstPtr, _, _ in
                    dstPtr.update(from: srcPtr.advanced(by: srcOffset), count: dModel)
                }
            }

            var symbolsThisStep = 0
            while symbolsThisStep < Self.maxSymbolsPerStep {
                // Read the current prediction embedding. At t=0 (and after
                // utterance reset) this is the zero vector; otherwise it's
                // the embedding produced by the most recent non-blank emit.
                let predEmb = await prediction.currentEmbedding()

                // Run the joiner. This is the inner-loop hot path — one call
                // per (t, symbol) pair, up to T * maxSymbolsPerStep per chunk.
                let logits = try await joiner.forward(enc: encStep, pred: predEmb)

                // Argmax over the vocab axis. `nonisolated` on the joiner
                // actor — no extra `await`.
                let token = joiner.argmax(logits)

                if token == Self.blankId {
                    // Blank: stop emitting at this t, advance to t+1. This is
                    // the only path out of the inner loop other than hitting
                    // `maxSymbolsPerStep`.
                    break
                }

                // Non-blank: commit the token to the accumulator and advance
                // the prediction LSTM. The LSTM step caches the new embedding
                // into prediction.lastEmbedding; the next iteration of this
                // inner loop will read it via currentEmbedding().
                emittedTokens.append(token)
                try await prediction.step(token: token)
                symbolsThisStep += 1
            }
            // If symbolsThisStep == maxSymbolsPerStep we silently advance t.
            // That's the stutter-safety bound doing its job; no warning is
            // emitted because Nemotron at 320 ms chunks rarely hits it on
            // clean speech (Phase 1 measures this — if it fires routinely,
            // suspect a parity bug, not a model issue).
        }

        // Slice out the tokens emitted by THIS call.
        return Array(emittedTokens[startIndex..<emittedTokens.count])
    }

    // MARK: - Reset

    /// Clear the utterance-wide accumulator and reset the prediction LSTM at
    /// utterance boundary.
    ///
    /// Per PLAN.md §4 "Utterance boundary" and Anti-pattern 11: must be called
    /// alongside `EncoderSession.reset()` on the same boundary — never
    /// independently. The decoder does NOT call `EncoderSession.reset()`
    /// because it doesn't hold a reference to the encoder; that's the
    /// `NemotronStreamingEngine` actor's job (Phase F).
    ///
    /// Resetting the joiner is a no-op (stateless), which is why no
    /// `joiner.reset()` call appears here.
    public func reset() async {
        emittedTokens.removeAll(keepingCapacity: true)
        await prediction.reset()
        // CPU decoder state resets to zeros + the blank-as-SOS step, matching
        // the engine's prediction-session seeding.
        cpu?.reset()
    }
}
