// Cohere Transcribe 2B — attention encoder-decoder (AED) engine.
//
// Unlike the Nemotron / Parakeet transducers, Cohere is a cross-attention
// encoder-decoder: a full-context Fast-Conformer encoder feeds an autoregressive
// transformer decoder. The decoder runs one token per step with an EXTERNAL KV
// cache (self-attention K/V threaded in/out as plain tensors; cross-attention
// recomputed each step from the encoder output). Both `.aimodel` functions are
// stateless — the host owns the caches and builds the attention masks.
//
//   encode(input_features[1,128,3500], length[1]) -> encoder_hidden_states[1,438,1024]
//   decode_step(input_id, position_id, encoder_hidden_states,
//               self_attention_mask[1,1,1,108], cross_attention_mask[1,1,1,438],
//               k_cache[8,1,8,108,128], v_cache[8,1,8,108,128])
//       -> logits[1,16384], k_cache_out, v_cache_out

import Accelerate
import CoreAI
import Foundation

public actor CohereTranscribeEngine {
    private let encodeFn: InferenceFunction
    private let decodeFn: InferenceFunction
    private let tokenizer: SentencePieceTokenizer
    private let melExtractor: MelExtractor

    // Cohere Transcribe constants.
    private static let melFrames = 3500          // fixed 35 s encoder window
    private static let nMels = 128
    private static let encDim = 1024
    private static let tEnc = 438                // encoder output frames (3500 -> /8)
    private static let nLayers = 8
    private static let nHeads = 8
    private static let headDim = 128
    private static let maxCacheLen = 108         // max decode steps
    private static let vocab = 16384             // SentencePiece vocab size
    private static let startToken: Int32 = 13764 // decoder_start_token_id
    private static let eosToken: Int32 = 3       // <|endoftext|>
    private static let maskNeg: Float = -1e9
    private static let traceDecode = ProcessInfo.processInfo.environment["COHERE_TRACE"] != nil

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
        guard let dec = try model.loadFunction(named: "decode_step") else {
            throw ASRError.missingFunction("decode_step")
        }
        self.encodeFn = enc
        self.decodeFn = dec
        let tokenizer = try SentencePieceTokenizer(piecesURL: piecesURL)
        guard tokenizer.count == Self.vocab else {
            throw ASRError.modelLoadFailed(
                "vocab mismatch: \(piecesURL.lastPathComponent) has \(tokenizer.count) pieces, "
                + "expected \(Self.vocab) for Cohere — wrong --pieces / pieces.txt?")
        }
        self.tokenizer = tokenizer
        // Cohere's preprocessor is the same NeMo per-feature / STFT-center mel as
        // Parakeet, so center the 400-sample Hann in the 512 FFT (offset 56).
        self.melExtractor = MelExtractor(frameStartOffset: (512 - 400) / 2)
    }

    /// Encoder valid output length for `realFrames` mel frames — three stride-2
    /// subsampling convs, matching the model's `_infer_encoder_lengths_from_raw`.
    /// Internal (not private) so it can be unit-tested.
    static func encLen(_ realFrames: Int) -> Int {
        var l = realFrames
        for _ in 0..<3 { l = (l - 1) / 2 + 1 }
        return max(1, min(l, tEnc))
    }

    public func transcribe(audio: [Float]) async throws -> String {
        await melExtractor.reset()

        // Mel -> [1,128,3500], per-feature normalized over the real frames.
        let melFlat = await melExtractor.extract(samples: audio)  // [128 × T]
        let realT = melFlat.count / Self.nMels
        let useT = min(realT, Self.melFrames)
        let normFlat = perFeatureNormalize(melFlat, nMels: Self.nMels, frames: realT)

        var melND = NDArray(shape: [1, Self.nMels, Self.melFrames], scalarType: .float32)
        var melMV = melND.mutableView(as: Float.self)
        melMV.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: 0, count: Self.nMels * Self.melFrames)
            for m in 0..<Self.nMels {
                let src = m * realT, dst = m * Self.melFrames
                for t in 0..<useT { ptr[dst + t] = normFlat[src + t] }
            }
        }
        var lenND = NDArray(shape: [1], scalarType: .int32)
        lenND.setInt32(at: 0, to: Int32(useT))

        // Encode once.
        var encOut = try await encodeFn.run(
            inputs: ["input_features": melND, "length": lenND],
            states: InferenceFunction.MutableViews()
        )
        guard let enc = encOut.remove("encoder_hidden_states")?.ndArray else {
            throw ASRError.missingOutput("encoder_hidden_states")
        }

        let emitted = try await decode(enc: enc, encLen: Self.encLen(useT))
        return tokenizer.decode(emitted)
    }

    /// Greedy autoregressive decode with an external self-attention KV cache.
    private func decode(enc: NDArray, encLen: Int) async throws -> [Int32] {
        // Caches: [L,1,H,M,D], zero-initialized.
        let cacheShape = [Self.nLayers, 1, Self.nHeads, Self.maxCacheLen, Self.headDim]
        var kCache = NDArray(shape: cacheShape, scalarType: .float32)
        var vCache = NDArray(shape: cacheShape, scalarType: .float32)
        kCache.zeroFloat32()
        vCache.zeroFloat32()

        // Cross-attention mask [1,1,1,438]: 0 for valid encoder frames, -1e9 else.
        var crossMask = NDArray(shape: [1, 1, 1, Self.tEnc], scalarType: .float32)
        var crossMV = crossMask.mutableView(as: Float.self)
        crossMV.withUnsafeMutablePointer { ptr, _, _ in
            for j in 0..<Self.tEnc { ptr[j] = j < encLen ? 0 : Self.maskNeg }
        }

        var selfMask = NDArray(shape: [1, 1, 1, Self.maxCacheLen], scalarType: .float32)
        var inputND = NDArray(shape: [1, 1], scalarType: .int32)
        var posND = NDArray(shape: [1, 1], scalarType: .int32)

        var emitted: [Int32] = []
        var prev = Self.startToken
        var hitEOS = false
        for pos in 0..<Self.maxCacheLen {
            inputND.setInt32(at: 0, to: prev)
            posND.setInt32(at: 0, to: Int32(pos))
            // Self-attention mask: attend to cache positions 0...pos.
            var selfMV = selfMask.mutableView(as: Float.self)
            selfMV.withUnsafeMutablePointer { ptr, _, _ in
                for j in 0..<Self.maxCacheLen { ptr[j] = j <= pos ? 0 : Self.maskNeg }
            }

            var out = try await decodeFn.run(
                inputs: [
                    "input_id": inputND, "position_id": posND,
                    "encoder_hidden_states": enc,
                    "self_attention_mask": selfMask, "cross_attention_mask": crossMask,
                    "k_cache": kCache, "v_cache": vCache,
                ],
                states: InferenceFunction.MutableViews()
            )
            guard let logits = out.remove("logits")?.ndArray,
                  let kOut = out.remove("k_cache_out")?.ndArray,
                  let vOut = out.remove("v_cache_out")?.ndArray else {
                throw ASRError.missingOutput("decode_step outputs")
            }
            kCache = kOut
            vCache = vOut

            var maxV: Float = 0
            var maxI: vDSP_Length = 0
            logits.view(as: Float.self).withUnsafePointer { ptr, _, _ in
                vDSP_maxvi(ptr, 1, &maxV, &maxI, vDSP_Length(logits.elementCount))
            }
            let next = Int32(maxI)
            if Self.traceDecode {
                FileHandle.standardError.write(Data("[cohere] pos=\(pos) tok=\(next)\n".utf8))
            }
            if next == Self.eosToken { hitEOS = true; break }
            emitted.append(next)
            prev = next
        }
        // The decoder's self-attention cache is a fixed `maxCacheLen` slots (the
        // model's baked CACHE_LEN). A clip whose transcript exceeds that token
        // budget runs out of cache before emitting EOS, so the tail is silently
        // dropped. Surface it rather than returning a quietly-truncated string.
        if !hitEOS {
            FileHandle.standardError.write(Data((
                "[cohere] warning: hit the \(Self.maxCacheLen)-token decode limit without EOS — "
                + "transcript is likely TRUNCATED (clip exceeds the model's per-window token budget).\n").utf8))
        }
        return emitted
    }
}
