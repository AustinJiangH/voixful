// Parakeet TDT 0.6B engine — full-context (batch) ASR.
//
// Parakeet differs from the streaming Nemotron path in two ways the rest of the
// package is NOT built for, so this is a separate engine:
//   1. Full-context encoder: the whole clip's mel goes in ONE `encode` call
//      (no streaming caches). The convert pipeline fixes the mel length at 1000
//      frames; we pad/truncate to that.
//   2. TDT (Token-and-Duration Transducer) joint: its output is
//      [1,1,1, vocab+1 + numDurations] = token logits (8193) followed by
//      duration logits (5). Greedy decode argmaxes BOTH and advances the encoder
//      time index by the predicted DURATION, not always +1.
//
// Reuses `PredictionSession` + `JoinerSession` (same function/state names); the
// `encode` function is called directly because `EncoderSession` is Nemotron's
// cache-aware shape.

import Accelerate
import CoreAI
import Foundation

public actor ParakeetEngine {
    private let encodeFn: InferenceFunction
    private let prediction: PredictionSession
    private let joiner: JoinerSession
    private let tokenizer: Tokenizer
    private let melExtractor: MelExtractor

    // Parakeet TDT 0.6B constants (probed from the checkpoint).
    private static let melFrames = 1000          // fixed encoder mel length (convert.py)
    private static let nMels = 128
    private static let dModel = 1024
    private static let vocab = 8192
    private static let blankId: Int32 = 8192     // vocab is blank id (token logits = vocab+1)
    private static let tokenLogits = 8193
    private static let durations: [Int] = [0, 1, 2, 3, 4]
    private static let maxSymbolsPerStep = 10
    private static let traceDecode = ProcessInfo.processInfo.environment["PARAKEET_TRACE"] != nil

    public init(modelURL: URL, piecesURL: URL) async throws {
        let model: AIModel
        do {
            if let cached = try AIModelCache.default.model(for: modelURL, options: .default) {
                model = cached
            } else {
                model = try await AIModel.specialize(
                    contentsOf: modelURL, options: .default,
                    cache: AIModelCache.default, cachePolicy: .persistent
                )
            }
        } catch {
            throw ASRError.modelLoadFailed("specialize(\(modelURL.path)): \(error)")
        }
        guard let enc = try model.loadFunction(named: "encode") else {
            throw ASRError.missingFunction("encode")
        }
        self.encodeFn = enc
        self.prediction = try await PredictionSession(model: model)
        self.joiner = try await JoinerSession(model: model)
        self.tokenizer = try Tokenizer(vocabURL: piecesURL)
        // Parakeet's preprocessor is `torch.stft(center=True)` with win<n_fft, so
        // center the 400-sample Hann in the 512 frame — offset (512-400)/2 = 56.
        self.melExtractor = MelExtractor(frameStartOffset: (512 - 400) / 2)
    }

    /// Transcribe a complete 16 kHz mono clip (full-context, one shot).
    public func transcribe(audio: [Float]) async throws -> String {
        await prediction.reset()
        await melExtractor.reset()

        // SOS seed: NeMo seeds the prediction network with g_0 = LSTM(zero_embed,
        // zero_state), NOT a literal zero vector. The prediction embedding uses
        // `padding_idx = blank`, so feeding the blank token produces exactly that
        // zero-embedding input → the correct g_0. Without this the TDT joint emits
        // blank through the start of the utterance (the LM prior never engages).
        try await prediction.step(token: Self.blankId)

        // Mel → fixed [1, 128, 1000] NDArray (pad/truncate the time axis).
        let melFlat = await melExtractor.extract(samples: audio)  // row-major [128 × T]
        let realT = melFlat.count / Self.nMels
        let useT = min(realT, Self.melFrames)

        // Parakeet's preprocessor uses `normalize: per_feature` (confirmed in the
        // checkpoint config) — each mel bin is standardized to zero mean / unit std
        // over the utterance's valid frames, matching NeMo `normalize_batch`. The
        // Nemotron path uses `normalize: NA`, so MelExtractor doesn't do this and
        // we apply it here. Without it the encoder sees out-of-distribution input
        // and the TDT decoder collapses to the prediction network's LM prior.
        let normFlat = perFeatureNormalize(melFlat, nMels: Self.nMels, frames: realT)

        var melND = NDArray(shape: [1, Self.nMels, Self.melFrames], scalarType: .float32)
        var mv = melND.mutableView(as: Float.self)
        mv.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: 0, count: Self.nMels * Self.melFrames)
            for m in 0..<Self.nMels {
                let srcRow = m * realT
                let dstRow = m * Self.melFrames
                for t in 0..<useT { ptr[dstRow + t] = normFlat[srcRow + t] }
            }
        }
        var lenND = NDArray(shape: [1], scalarType: .int32)
        lenND.setInt32(at: 0, to: Int32(useT))

        // Full-context encode (no states).
        var outputs = try await encodeFn.run(
            inputs: ["mel": melND, "mel_length": lenND],
            states: InferenceFunction.MutableViews()
        )
        guard let encOut = outputs.remove("enc_out")?.ndArray else {
            throw ASRError.missingOutput("enc_out")
        }

        let emitted = try await tdtDecode(encOut: encOut, realFrames: useT)
        if Self.traceDecode {
            FileHandle.standardError.write(Data(
                "[parakeet] realT=\(realT) enc=\(encOut.shape) emitted=\(emitted.count)\n".utf8
            ))
        }
        return await tokenizer.decode(tokens: emitted, stripLangTags: true)
    }

    /// TDT greedy decode over the encoder time axis.
    private func tdtDecode(encOut: NDArray, realFrames: Int) async throws -> [Int32] {
        let tOut = encOut.shape[1]
        // Encoder subsamples mel by 8; don't decode into the zero-padded tail.
        let frames = min(tOut, (realFrames + 7) / 8 + 1)
        var encStep = NDArray(shape: [1, 1, Self.dModel], scalarType: .float32)
        var emitted: [Int32] = []

        var t = 0
        while t < frames {
            // Copy enc_out[:, t, :] into the scratch step.
            encOut.view(as: Float.self).withUnsafePointer { src, _, strides in
                let off = t * strides[1]
                var dv = encStep.mutableView(as: Float.self)
                dv.withUnsafeMutablePointer { dst, _, _ in
                    dst.update(from: src.advanced(by: off), count: Self.dModel)
                }
            }

            var symbols = 0
            while true {
                let predEmb = await prediction.currentEmbedding()
                let logits = try await joiner.forward(enc: encStep, pred: predEmb)

                // Argmax over the borrowed logits view — no per-call array copy.
                // Token argmax (8193 elems) via vDSP SIMD; duration argmax (5
                // elems) by hand over the tail. logits is [1,1,1, vocab+1+durs].
                var tok: Int32 = 0
                var durIdx = 0
                logits.view(as: Float.self).withUnsafePointer { ptr, _, _ in
                    var maxV: Float = 0
                    var maxI: vDSP_Length = 0
                    vDSP_maxvi(ptr, 1, &maxV, &maxI, vDSP_Length(Self.tokenLogits))
                    tok = Int32(maxI)
                    var durV: Float = -.infinity
                    for j in 0..<Self.durations.count {
                        let v = ptr[Self.tokenLogits + j]
                        if v > durV { durV = v; durIdx = j }
                    }
                }
                let skip = Self.durations[durIdx]

                if tok != Self.blankId {
                    emitted.append(tok)
                    try await prediction.step(token: tok)
                    symbols += 1
                }

                if Self.traceDecode {
                    FileHandle.standardError.write(Data("[tdt] t=\(t) tok=\(tok) skip=\(skip)\n".utf8))
                }

                if skip > 0 {
                    t += skip
                    break
                }
                // skip == 0: stay on the same frame to emit more, unless we hit
                // a blank or the per-frame cap — then force progress.
                if tok == Self.blankId || symbols >= Self.maxSymbolsPerStep {
                    t += 1
                    break
                }
            }
        }
        return emitted
    }
}
