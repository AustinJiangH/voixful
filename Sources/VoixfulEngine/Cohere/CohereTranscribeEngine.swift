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
    private let model: AIModel
    /// Model instance the decoder-side functions load from. Same as `model`
    /// unless `VOIXFUL_COHERE_DECODER_ASSET` points at the decoder-only
    /// companion asset (see convert.py --decoder-asset), which is the only way
    /// to give the decoder its own compute-unit placement
    /// (`VOIXFUL_COMPUTE_DECODE`) — specialization is whole-asset.
    private let decodeModel: AIModel
    /// Function names present in the asset. Functions are loaded LAZILY (see
    /// `fn(_:)`) — `loadFunction` specializes the graph, and eagerly loading
    /// every window/decoder variant materializes a specialized weight set per
    /// function (measured ~52 GB peak RSS on the 2B fp16 with 4 windows).
    private let available: Set<String>
    /// Smaller fixed-window encoder variants (`encode_<frames>`), ascending.
    /// Short clips pick the smallest window that fits instead of paying the
    /// full 35 s / 48-layer encode on zero padding; their output is zero-padded
    /// back up to the decoder's fixed 438 cross-attention frames (the
    /// cross-attention mask hides the padding). Empty for single-window assets.
    private let encodeWindows: [(frames: Int, name: String)]
    /// Lazily loaded functions, keyed by name.
    private var loaded: [String: InferenceFunction] = [:]
    /// In-flight background loads (prewarm). `fn(_:)` awaits these instead of
    /// duplicating a load — loadFunction on the 2B palettized encoder costs
    /// ~1.1 s of weight materialization per function.
    private var pending: [String: Task<InferenceFunction?, Never>] = [:]
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
            if let cached = try AIModelCache.default.model(for: modelURL, options: specializationOptions()) {
                model = cached
            } else {
                model = try await AIModel.specialize(
                    contentsOf: modelURL, options: specializationOptions(),
                    cache: AIModelCache.default, cachePolicy: .persistent
                )
            }
        } catch {
            throw ASRError.modelLoadFailed("specialize(\(modelURL.path)): \(error)")
        }
        self.model = model
        if let decPath = ProcessInfo.processInfo.environment["VOIXFUL_COHERE_DECODER_ASSET"] {
            let decURL = URL(fileURLWithPath: decPath)
            let opts = decodeSpecializationOptions() ?? specializationOptions()
            do {
                if let cached = try AIModelCache.default.model(for: decURL, options: opts) {
                    self.decodeModel = cached
                } else {
                    self.decodeModel = try await AIModel.specialize(
                        contentsOf: decURL, options: opts,
                        cache: AIModelCache.default, cachePolicy: .persistent
                    )
                }
            } catch {
                throw ASRError.modelLoadFailed("decoder asset specialize(\(decPath)): \(error)")
            }
        } else {
            self.decodeModel = model
        }
        let names = try ModelBackend.assetFunctions(modelURL: modelURL).names
        guard names.contains("encode") else { throw ASRError.missingFunction("encode") }
        // Any one decode entrypoint suffices; lean assets ship only the cached
        // decoders (decode_x4 / decode_step_fused) — the stateless decode_step
        // is a --legacy-decoders export.
        let decodeFns: Set<String> = ["decode_x4", "decode_step_fused", "decode_step_cached", "decode_step"]
        guard !names.isDisjoint(with: decodeFns) else {
            throw ASRError.missingFunction("decode_x4|decode_step_fused|decode_step_cached|decode_step")
        }
        self.available = names
        self.encodeWindows = encodeWindowNames(names)
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

    /// Decoder-side function names load from `decodeModel` (the companion
    /// asset when configured); encoder windows from the primary `model`.
    private static let decoderFns: Set<String> = [
        "project_cross_kv", "decode_x4", "decode_step_fused", "decode_step_cached", "decode_step",
    ]
    private func modelFor(_ name: String) -> AIModel {
        Self.decoderFns.contains(name) ? decodeModel : model
    }

    /// Load (and cache) a function by name — lazy so only the window/decoder
    /// variant a clip actually uses gets specialized. Awaits an in-flight
    /// prewarm load of the same name rather than loading twice.
    private func fn(_ name: String) async throws -> InferenceFunction {
        if let f = loaded[name] { return f }
        if let t = pending.removeValue(forKey: name) {
            if let f = await t.value {
                loaded[name] = f
                return f
            }
            // Prewarm failed; fall through to a direct load for the real error.
        }
        guard let f = try modelFor(name).loadFunction(named: name) else {
            throw ASRError.missingFunction(name)
        }
        loaded[name] = f
        return f
    }

    /// Kick off background loads of the functions a session will most likely
    /// need, so their ~1.1 s (encoder window) weight materialization overlaps
    /// audio capture / app idle instead of the first `transcribe()`. Returns
    /// immediately. `VOIXFUL_PREWARM=0` disables. Only the smallest encoder
    /// window is prewarmed — window choice depends on clip length, and eagerly
    /// materializing every window is the ~52 GB fp16 lesson; on a wrong guess
    /// the right window loads lazily exactly as before.
    public func prewarm() {
        guard ProcessInfo.processInfo.environment["VOIXFUL_PREWARM"] != "0" else { return }
        var names: [String] = []
        if available.contains("project_cross_kv") { names.append("project_cross_kv") }
        for decode in ["decode_x4", "decode_step_fused", "decode_step_cached", "decode_step"]
        where available.contains(decode) {
            names.append(decode)
            break
        }
        names.append(encodeWindows.first?.name ?? "encode")
        for name in names where loaded[name] == nil && pending[name] == nil {
            let model = modelFor(name)
            pending[name] = Task.detached(priority: .utility) {
                (try? model.loadFunction(named: name)) ?? nil
            }
        }
    }

    /// Encoder valid output length for `realFrames` mel frames — three stride-2
    /// subsampling convs, matching the model's `_infer_encoder_lengths_from_raw`.
    /// Internal (not private) so it can be unit-tested.
    static func encLen(_ realFrames: Int) -> Int {
        var l = realFrames
        for _ in 0..<3 { l = (l - 1) / 2 + 1 }
        return max(1, min(l, tEnc))
    }

    public func transcribe(audio: [Float]) async throws -> Transcript {
        await melExtractor.reset()
        let t0 = CFAbsoluteTimeGetCurrent()

        // Mel -> fixed-window [1,128,W], per-feature normalized over the real
        // frames. Pick the smallest exported window that fits the clip so short
        // clips don't pay the full 35 s / 48-layer encode on zero padding.
        let melFlat = await melExtractor.extract(samples: audio)  // [128 × T]
        let tExtract = CFAbsoluteTimeGetCurrent()
        let realT = melFlat.count / Self.nMels
        let picked = pickEncodeWindow(encodeWindows, realFrames: realT)
        let window = picked?.frames ?? Self.melFrames
        let encode = try await fn(picked?.name ?? "encode")
        let tLoad = CFAbsoluteTimeGetCurrent()
        let useT = min(realT, window)
        let normFlat = perFeatureNormalize(melFlat, nMels: Self.nMels, frames: realT)

        var melND = NDArray(shape: [1, Self.nMels, window], scalarType: .float32)
        var melMV = melND.mutableView(as: Float.self)
        melMV.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: 0, count: Self.nMels * window)
            for m in 0..<Self.nMels {
                let src = m * realT, dst = m * window
                for t in 0..<useT { ptr[dst + t] = normFlat[src + t] }
            }
        }
        var lenND = NDArray(shape: [1], scalarType: .int32)
        lenND.setInt32(at: 0, to: Int32(useT))
        let tMel = CFAbsoluteTimeGetCurrent()

        // Encode once.
        var encOut = try await encode.run(
            inputs: ["input_features": melND, "length": lenND],
            states: InferenceFunction.MutableViews()
        )
        guard var enc = encOut.remove("encoder_hidden_states")?.ndArray else {
            throw ASRError.missingOutput("encoder_hidden_states")
        }
        // A smaller window emits [1, encLen(window), 1024]; the decoder is traced
        // against [1, 438, 1024]. Zero-pad the tail — the cross-attention mask
        // (valid only below encLen(useT)) hides it from the decoder.
        if enc.shape[1] < Self.tEnc {
            let validFrames = enc.shape[1]
            var padded = NDArray(shape: [1, Self.tEnc, Self.encDim], scalarType: .float32)
            padded.zeroFloat32()
            enc.view(as: Float.self).withUnsafePointer { src, _, _ in
                var dv = padded.mutableView(as: Float.self)
                dv.withUnsafeMutablePointer { dst, _, _ in
                    dst.update(from: src, count: validFrames * Self.encDim)
                }
            }
            enc = padded
        }
        let tEnc = CFAbsoluteTimeGetCurrent()

        let emitted: [Int32]
        let hasProject = available.contains("project_cross_kv")
        if hasProject, available.contains("decode_x4") {
            emitted = try await decodeX4(
                project: try await fn("project_cross_kv"), step: try await fn("decode_x4"),
                enc: enc, encLen: Self.encLen(useT))
        } else if hasProject, available.contains("decode_step_fused") {
            emitted = try await decodeFused(
                project: try await fn("project_cross_kv"), step: try await fn("decode_step_fused"),
                enc: enc, encLen: Self.encLen(useT))
        } else if hasProject, available.contains("decode_step_cached") {
            emitted = try await decodeCached(
                project: try await fn("project_cross_kv"), step: try await fn("decode_step_cached"),
                enc: enc, encLen: Self.encLen(useT))
        } else {
            emitted = try await decode(step: try await fn("decode_step"), enc: enc, encLen: Self.encLen(useT))
        }
        if Self.traceDecode {
            let tDec = CFAbsoluteTimeGetCurrent()
            let steps = emitted.count + 1
            let audioS: Double = Double(audio.count) / 16_000
            let melMS: Double = (tExtract - t0) * 1000
            let loadMS: Double = (tLoad - tExtract) * 1000
            let packMS: Double = (tMel - tLoad) * 1000
            let encMS: Double = (tEnc - tMel) * 1000
            let decMS: Double = (tDec - tEnc) * 1000
            let line = String(
                format: "[cohere/timing] audio=%.1fs realT=%d useT=%d (window=%d) | "
                    + "mel=%.0fms load=%.0fms pack=%.0fms encode=%.0fms decode=%.0fms (%d steps, %.1fms/step)\n",
                audioS, realT, useT, window, melMS, loadMS, packMS, encMS, decMS, steps,
                decMS / Double(steps))
            FileHandle.standardError.write(Data(line.utf8))
        }
        // Batch AED decode produces one final transcript; segments aren't surfaced
        // and the language is model-auto-detected.
        return Transcript(text: tokenizer.decode(emitted), segments: [], language: nil)
    }

    /// Greedy decode in blocks of four: each `decode_x4` call runs four unrolled
    /// decoder steps in-graph (argmax feeding the next embedding), returning a
    /// [1,4] int32 token block. The host truncates at EOS. One dispatch per
    /// four tokens instead of per token.
    private func decodeX4(
        project: InferenceFunction, step: InferenceFunction,
        enc: NDArray, encLen: Int
    ) async throws -> [Int32] {
        let kvShape = [Self.nLayers, 1, Self.nHeads, Self.maxCacheLen, Self.headDim]
        var kCache = NDArray(shape: kvShape, scalarType: .float32)
        var vCache = NDArray(shape: kvShape, scalarType: .float32)
        kCache.zeroFloat32()
        vCache.zeroFloat32()

        var projOut = try await project.run(
            inputs: ["encoder_hidden_states": enc],
            states: InferenceFunction.MutableViews())
        guard let crossK = projOut.remove("cross_k")?.ndArray,
              let crossV = projOut.remove("cross_v")?.ndArray else {
            throw ASRError.missingOutput("project_cross_kv outputs")
        }

        var encLenND = NDArray(shape: [1], scalarType: .int32)
        encLenND.setInt32(at: 0, to: Int32(encLen))
        var inputND = NDArray(shape: [1, 1], scalarType: .int32)
        var posND = NDArray(shape: [1, 1], scalarType: .int32)

        var emitted: [Int32] = []
        var prev = Self.startToken
        var hitEOS = false
        var pos = 0
        blocks: while pos < Self.maxCacheLen {
            inputND.setInt32(at: 0, to: prev)
            posND.setInt32(at: 0, to: Int32(pos))
            var states = InferenceFunction.MutableViews()
            states.insert(&kCache, for: "k_cache")
            states.insert(&vCache, for: "v_cache")
            var out = try await step.run(
                inputs: [
                    "input_id": inputND, "position_id": posND, "enc_len": encLenND,
                    "cross_k": crossK, "cross_v": crossV,
                ],
                states: states
            )
            guard let block = out.remove("tokens")?.ndArray else {
                throw ASRError.missingOutput("decode_x4 tokens")
            }
            let n = block.elementCount
            var toks = [Int32](repeating: 0, count: n)
            block.view(as: Int32.self).withUnsafePointer { ptr, _, _ in
                for j in 0..<n { toks[j] = ptr[j] }
            }
            if Self.traceDecode {
                FileHandle.standardError.write(Data("[cohere] pos=\(pos) block=\(toks)\n".utf8))
            }
            for tok in toks {
                if tok == Self.eosToken { hitEOS = true; break blocks }
                emitted.append(tok)
            }
            prev = toks[n - 1]
            pos += n
        }
        if !hitEOS {
            FileHandle.standardError.write(Data((
                "[cohere] warning: hit the \(Self.maxCacheLen)-token decode limit without EOS — "
                + "transcript is likely TRUNCATED (clip exceeds the model's per-window token budget).\n").utf8))
        }
        return emitted
    }

    /// Greedy decode via the fused decoder: masks built in-graph, argmax token
    /// returned directly. Per-step host I/O = input_id + position_id + enc_len
    /// (+ the reused cross K/V handles) in, one int32 token out.
    private func decodeFused(
        project: InferenceFunction, step: InferenceFunction,
        enc: NDArray, encLen: Int
    ) async throws -> [Int32] {
        let kvShape = [Self.nLayers, 1, Self.nHeads, Self.maxCacheLen, Self.headDim]
        var kCache = NDArray(shape: kvShape, scalarType: .float32)
        var vCache = NDArray(shape: kvShape, scalarType: .float32)
        kCache.zeroFloat32()
        vCache.zeroFloat32()

        var projOut = try await project.run(
            inputs: ["encoder_hidden_states": enc],
            states: InferenceFunction.MutableViews())
        guard let crossK = projOut.remove("cross_k")?.ndArray,
              let crossV = projOut.remove("cross_v")?.ndArray else {
            throw ASRError.missingOutput("project_cross_kv outputs")
        }

        var encLenND = NDArray(shape: [1], scalarType: .int32)
        encLenND.setInt32(at: 0, to: Int32(encLen))
        var inputND = NDArray(shape: [1, 1], scalarType: .int32)
        var posND = NDArray(shape: [1, 1], scalarType: .int32)

        var emitted: [Int32] = []
        var prev = Self.startToken
        var hitEOS = false
        for pos in 0..<Self.maxCacheLen {
            inputND.setInt32(at: 0, to: prev)
            posND.setInt32(at: 0, to: Int32(pos))
            var states = InferenceFunction.MutableViews()
            states.insert(&kCache, for: "k_cache")
            states.insert(&vCache, for: "v_cache")
            var out = try await step.run(
                inputs: [
                    "input_id": inputND, "position_id": posND, "enc_len": encLenND,
                    "cross_k": crossK, "cross_v": crossV,
                ],
                states: states
            )
            guard let tokenND = out.remove("token")?.ndArray else {
                throw ASRError.missingOutput("decode_step_fused token")
            }
            let next = tokenND.view(as: Int32.self).withUnsafePointer { ptr, _, _ in ptr[0] }
            if Self.traceDecode {
                FileHandle.standardError.write(Data("[cohere] pos=\(pos) tok=\(next)\n".utf8))
            }
            if next == Self.eosToken { hitEOS = true; break }
            emitted.append(next)
            prev = next
        }
        if !hitEOS {
            FileHandle.standardError.write(Data((
                "[cohere] warning: hit the \(Self.maxCacheLen)-token decode limit without EOS — "
                + "transcript is likely TRUNCATED (clip exceeds the model's per-window token budget).\n").utf8))
        }
        return emitted
    }

    /// Greedy decode via the cached-decoder pair: cross-attention K/V projected
    /// once into host-owned state buffers, self-attention cache mutated in place.
    /// Per-step traffic is just the two ids + two masks (vs ~14 MB threaded
    /// tensors in the stateless `decode`).
    private func decodeCached(
        project: InferenceFunction, step: InferenceFunction,
        enc: NDArray, encLen: Int
    ) async throws -> [Int32] {
        // Self-attention cache: host-owned state buffers mutated in place.
        let kvShape = [Self.nLayers, 1, Self.nHeads, Self.maxCacheLen, Self.headDim]
        var kCache = NDArray(shape: kvShape, scalarType: .float32)
        var vCache = NDArray(shape: kvShape, scalarType: .float32)
        kCache.zeroFloat32()
        vCache.zeroFloat32()

        // Project every layer's cross-attention K/V once for this utterance.
        var projOut = try await project.run(
            inputs: ["encoder_hidden_states": enc],
            states: InferenceFunction.MutableViews())
        guard let crossK = projOut.remove("cross_k")?.ndArray,
              let crossV = projOut.remove("cross_v")?.ndArray else {
            throw ASRError.missingOutput("project_cross_kv outputs")
        }

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
            var selfMV = selfMask.mutableView(as: Float.self)
            selfMV.withUnsafeMutablePointer { ptr, _, _ in
                for j in 0..<Self.maxCacheLen { ptr[j] = j <= pos ? 0 : Self.maskNeg }
            }

            var states = InferenceFunction.MutableViews()
            states.insert(&kCache, for: "k_cache")
            states.insert(&vCache, for: "v_cache")
            var out = try await step.run(
                inputs: [
                    "input_id": inputND, "position_id": posND,
                    "self_attention_mask": selfMask, "cross_attention_mask": crossMask,
                    "cross_k": crossK, "cross_v": crossV,
                ],
                states: states
            )
            guard let logits = out.remove("logits")?.ndArray else {
                throw ASRError.missingOutput("decode_step_cached logits")
            }

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
        if !hitEOS {
            FileHandle.standardError.write(Data((
                "[cohere] warning: hit the \(Self.maxCacheLen)-token decode limit without EOS — "
                + "transcript is likely TRUNCATED (clip exceeds the model's per-window token budget).\n").utf8))
        }
        return emitted
    }

    /// Greedy autoregressive decode with an external self-attention KV cache.
    private func decode(step decodeFn: InferenceFunction, enc: NDArray, encLen: Int) async throws -> [Int32] {
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
