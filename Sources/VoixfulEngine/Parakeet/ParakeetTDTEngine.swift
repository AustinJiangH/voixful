// Parakeet TDT 0.6B engine — full-context (batch) ASR.
//
// Parakeet differs from the streaming Nemotron path in two ways the rest of the
// package is NOT built for, so this is a separate engine:
//   1. Full-context encoder: the whole clip's mel goes in ONE `encode` call
//      (no streaming caches). The convert pipeline traces a fixed mel length
//      (the largest window, up to 3000 frames = 30 s, recorded in
//      parakeet.json); we pad/truncate each clip to the window it fits.
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

public actor ParakeetTDTEngine {
    private let model: AIModel
    /// Smaller fixed-window encoder variants (`encode_<frames>`), ascending by
    /// window. The traced encoder always runs its full window, so short clips in
    /// the 1000-frame window burn compute on zero padding — `transcribe` picks
    /// the smallest window that fits the clip. Empty for single-window assets.
    /// Functions load LAZILY — each load specializes its own weight set, so
    /// eager loading of every window multiplies resident weight memory.
    private let encodeWindows: [(frames: Int, name: String)]
    /// Lazily loaded encode functions, keyed by name.
    private var loaded: [String: InferenceFunction] = [:]
    /// Function names present in the asset (fused-decode capability check).
    private let available: Set<String>
    /// CPU predict/join decoder, present when the `<asset>.transducer.json/bin`
    /// sidecar exists (see convert/parakeet-tdt dump_transducer.py). Bypasses
    /// the ~0.4 ms/dispatch Core AI floor for the tiny decode-side math.
    private let cpuDecoder: CPUTransducerDecoder?
    private let prediction: PredictionSession
    private let joiner: JoinerSession
    private let tokenizer: SentencePieceTokenizer
    private let melExtractor: MelExtractor

    // Parakeet TDT 0.6B constants (probed from the checkpoint).
    /// Largest encoder window = the bare `encode` function's mel length, and the
    /// `project_enc` trace length (= melFrames / 8). Read from the asset's
    /// `parakeet.json` (new assets encode up to 3000 frames = 30 s); assets
    /// predating the manifest fall back to the historical 1000 / 125.
    private let melFrames: Int
    private let projT: Int

    /// Largest single-pass audio: `melFrames` × 160-sample hop (10 s at 1000
    /// frames, 30 s at 3000). Longer audio is segmented by the callers (see
    /// VoixfulFullContext `LongForm`), never truncated in normal use.
    public nonisolated var maxAudioSamples: Int { melFrames * 160 }
    private static let nMels = 128
    private static let dModel = 1024
    private static let durations: [Int] = [0, 1, 2, 3, 4]
    // Vocab-dependent ids come from the asset's pieces.txt: v3 ships 8192
    // SentencePiece tokens, v2 ships 1024. blank = vocab (embedding padding_idx);
    // token logits = vocab + 1; joint output = tokenLogits + 5 durations.
    private let vocab: Int
    private let blankId: Int32
    private let tokenLogits: Int
    private static let maxSymbolsPerStep = 10
    private static let traceDecode = ProcessInfo.processInfo.environment["PARAKEET_TRACE"] != nil

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
        let names = try ModelBackend.assetFunctions(modelURL: modelURL).names
        guard names.contains("encode") else { throw ASRError.missingFunction("encode") }
        // Largest window + projector length from the asset manifest; older
        // assets without parakeet.json keep the historical 1000-frame (10 s)
        // window and T=125 projector.
        if let data = try? Data(contentsOf: modelURL.appendingPathComponent("parakeet.json")),
           let m = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let mf = m["mel_frames"] as? Int, let pt = m["proj_t"] as? Int {
            self.melFrames = mf
            self.projT = pt
        } else {
            self.melFrames = 1000
            self.projT = 125
        }
        self.encodeWindows = encodeWindowNames(names)
        self.available = names
        self.cpuDecoder = names.contains("project_enc")
            ? CPUTransducerDecoder(sidecarFor: modelURL) : nil
        // Optional separate placement for predict/join (see ComputeUnit.swift).
        var decodeModel = model
        if let opts = decodeSpecializationOptions() {
            do {
                decodeModel = try await AIModel(contentsOf: modelURL, options: opts)
            } catch {
                throw ASRError.modelLoadFailed(
                    "decode-model AIModel(contentsOf:) with VOIXFUL_COMPUTE_DECODE failed: \(error)")
            }
        }
        self.prediction = try await PredictionSession(model: decodeModel)
        self.joiner = try await JoinerSession(model: decodeModel)
        let tokenizer = try SentencePieceTokenizer(piecesURL: piecesURL)
        let pieceCount = tokenizer.count
        self.vocab = pieceCount
        self.blankId = Int32(pieceCount)
        self.tokenLogits = pieceCount + 1
        // The CPU-decode sidecar carries its own vocab; a mismatch with
        // pieces.txt means asset and sidecar are from different checkpoints —
        // fail here rather than decode garbage.
        if let cpu = cpuDecoder, cpu.tokenLogits != pieceCount + 1 {
            throw ASRError.modelLoadFailed(
                "sidecar/vocab mismatch: transducer sidecar has \(cpu.tokenLogits) token "
                + "logits but pieces.txt has \(pieceCount) pieces (expected \(pieceCount + 1))")
        }
        self.tokenizer = tokenizer
        // Parakeet's preprocessor is `torch.stft(center=True)` with win<n_fft, so
        // center the 400-sample Hann in the 512 frame — offset (512-400)/2 = 56.
        self.melExtractor = MelExtractor(frameStartOffset: (512 - 400) / 2)
    }

    /// In-flight background loads (prewarm); see `prewarm()`.
    private var pending: [String: Task<InferenceFunction?, Never>] = [:]

    /// Load (and cache) an encode function by name — lazy so only the window a
    /// clip actually uses gets specialized. Awaits an in-flight prewarm load
    /// of the same name rather than loading twice.
    private func fn(_ name: String) async throws -> InferenceFunction {
        if let f = loaded[name] { return f }
        if let t = pending.removeValue(forKey: name) {
            if let f = await t.value {
                loaded[name] = f
                return f
            }
        }
        guard let f = try model.loadFunction(named: name) else {
            throw ASRError.missingFunction(name)
        }
        loaded[name] = f
        return f
    }

    /// Background-load the smallest encoder window so its weight
    /// materialization overlaps audio capture instead of the first
    /// `transcribe()` (predict/join already load eagerly in init — they are
    /// tiny). Returns immediately. `VOIXFUL_PREWARM=0` disables.
    public func prewarm() {
        guard ProcessInfo.processInfo.environment["VOIXFUL_PREWARM"] != "0" else { return }
        let name = encodeWindows.first?.name ?? "encode"
        guard loaded[name] == nil, pending[name] == nil else { return }
        let model = self.model
        pending[name] = Task.detached(priority: .utility) {
            (try? model.loadFunction(named: name)) ?? nil
        }
    }

    /// Transcribe a complete 16 kHz mono clip (full-context, one shot).
    public func transcribe(audio: [Float]) async throws -> Transcript {
        await prediction.reset()
        await melExtractor.reset()

        // SOS seed: NeMo seeds the prediction network with g_0 = LSTM(zero_embed,
        // zero_state), NOT a literal zero vector. The prediction embedding uses
        // `padding_idx = blank`, so feeding the blank token produces exactly that
        // zero-embedding input → the correct g_0. Without this the TDT joint emits
        // blank through the start of the utterance (the LM prior never engages).
        try await prediction.step(token: blankId)
        let t0 = CFAbsoluteTimeGetCurrent()

        // Mel → fixed-window NDArray (pad/truncate the time axis). Pick the
        // smallest exported encoder window that fits the clip so short clips
        // don't pay the full 1000-frame encode on zero padding.
        let melFlat = await melExtractor.extract(samples: audio)  // row-major [128 × T]
        let realT = melFlat.count / Self.nMels
        let picked = pickEncodeWindow(encodeWindows, realFrames: realT)
        let window = picked?.frames ?? melFrames
        let encode = try await fn(picked?.name ?? "encode")
        let useT = min(realT, window)

        // Parakeet's preprocessor uses `normalize: per_feature` (confirmed in the
        // checkpoint config) — each mel bin is standardized to zero mean / unit std
        // over the utterance's valid frames, matching NeMo `normalize_batch`. The
        // Nemotron path uses `normalize: NA`, so MelExtractor doesn't do this and
        // we apply it here. Without it the encoder sees out-of-distribution input
        // and the TDT decoder collapses to the prediction network's LM prior.
        let normFlat = perFeatureNormalize(melFlat, nMels: Self.nMels, frames: realT)

        var melND = NDArray(shape: [1, Self.nMels, window], scalarType: .float32)
        var mv = melND.mutableView(as: Float.self)
        mv.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: 0, count: Self.nMels * window)
            for m in 0..<Self.nMels {
                let srcRow = m * realT
                let dstRow = m * window
                for t in 0..<useT { ptr[dstRow + t] = normFlat[srcRow + t] }
            }
        }
        var lenND = NDArray(shape: [1], scalarType: .int32)
        lenND.setInt32(at: 0, to: Int32(useT))

        // Full-context encode (no states).
        let tMel = CFAbsoluteTimeGetCurrent()
        var outputs = try await encode.run(
            inputs: ["mel": melND, "mel_length": lenND],
            states: InferenceFunction.MutableViews()
        )
        guard let encOut = outputs.remove("enc_out")?.ndArray else {
            throw ASRError.missingOutput("enc_out")
        }
        let tEnc = CFAbsoluteTimeGetCurrent()

        let emitted: [Int32]
        if let cpu = cpuDecoder {
            emitted = try await tdtDecodeCPU(cpu: cpu, encOut: encOut, realFrames: useT)
        } else if available.contains("project_enc"), available.contains("join_fused") {
            emitted = try await tdtDecodeFused(encOut: encOut, realFrames: useT)
        } else {
            emitted = try await tdtDecode(encOut: encOut, realFrames: useT)
        }
        if Self.traceDecode {
            let tDec = CFAbsoluteTimeGetCurrent()
            let line = String(
                format: "[parakeet/timing] audio=%.1fs realT=%d useT=%d (window=%d) | "
                    + "mel=%.0fms encode=%.0fms decode=%.0fms emitted=%d\n",
                Double(audio.count) / 16_000, realT, useT, window,
                (tMel - t0) * 1000, (tEnc - tMel) * 1000, (tDec - tEnc) * 1000, emitted.count)
            FileHandle.standardError.write(Data(line.utf8))
        }
        // Batch decode produces one final transcript; per-token time alignment
        // (segments) is not surfaced yet, and language is model-auto-detected.
        return Transcript(text: tokenizer.decode(emitted, stripLangTags: true),
                          segments: [], language: nil)
    }

    /// Joint hidden width (project_encoder / project_prednet output).
    private static let jointHidden = 640

    /// TDT greedy decode with predict/join fully on CPU (Accelerate): one
    /// `project_enc` dispatch for the whole clip, one 320 KB readback, then
    /// zero Core AI dispatches per step.
    private func tdtDecodeCPU(
        cpu: CPUTransducerDecoder, encOut: NDArray, realFrames: Int
    ) async throws -> [Int32] {
        let tOut = encOut.shape[1]
        let frames = min(tOut, (realFrames + 7) / 8 + 1)

        var encPad = NDArray(shape: [1, projT, Self.dModel], scalarType: .float32)
        var pv = encPad.mutableView(as: Float.self)
        pv.withUnsafeMutablePointer { dst, _, _ in
            dst.update(repeating: 0, count: projT * Self.dModel)
            encOut.view(as: Float.self).withUnsafePointer { src, _, _ in
                dst.update(from: src, count: min(tOut, projT) * Self.dModel)
            }
        }
        var projOutputs = try await fn("project_enc").run(
            inputs: ["enc_out": encPad],
            states: InferenceFunction.MutableViews()
        )
        guard let encProj = projOutputs.remove("enc_proj")?.ndArray else {
            throw ASRError.missingOutput("enc_proj")
        }

        cpu.reset()
        var emitted: [Int32] = []
        encProj.view(as: Float.self).withUnsafePointer { base, _, strides in
            var t = 0
            while t < frames {
                var symbols = 0
                while true {
                    let (tok, skip) = cpu.join(encProjT: base.advanced(by: t * strides[1]))
                    if tok != blankId {
                        emitted.append(tok)
                        cpu.step(token: tok)
                        symbols += 1
                    }
                    if skip > 0 { t += skip; break }
                    if tok == blankId || symbols >= Self.maxSymbolsPerStep { t += 1; break }
                }
            }
        }
        return emitted
    }

    /// TDT greedy decode via the fused pair: `project_enc` once per clip,
    /// then `join_fused` per step — both argmaxes in-graph, so per-step host
    /// I/O is 640 floats in and two int32 scalars out instead of a
    /// [1,1,1,8198] logits readback + host argmax.
    private func tdtDecodeFused(encOut: NDArray, realFrames: Int) async throws -> [Int32] {
        let tOut = encOut.shape[1]
        let frames = min(tOut, (realFrames + 7) / 8 + 1)

        // Zero-pad enc_out to the projector's fixed [1, 125, 1024] frame.
        var encPad = NDArray(shape: [1, projT, Self.dModel], scalarType: .float32)
        var pv = encPad.mutableView(as: Float.self)
        pv.withUnsafeMutablePointer { dst, _, _ in
            dst.update(repeating: 0, count: projT * Self.dModel)
            encOut.view(as: Float.self).withUnsafePointer { src, _, _ in
                dst.update(from: src, count: min(tOut, projT) * Self.dModel)
            }
        }
        var projOutputs = try await fn("project_enc").run(
            inputs: ["enc_out": encPad],
            states: InferenceFunction.MutableViews()
        )
        guard let encProj = projOutputs.remove("enc_proj")?.ndArray else {
            throw ASRError.missingOutput("enc_proj")
        }

        let joinFused = try await fn("join_fused")
        var encProjT = NDArray(shape: [1, 1, Self.jointHidden], scalarType: .float32)
        var emitted: [Int32] = []

        var t = 0
        while t < frames {
            encProj.view(as: Float.self).withUnsafePointer { src, _, strides in
                let off = t * strides[1]
                var dv = encProjT.mutableView(as: Float.self)
                dv.withUnsafeMutablePointer { dst, _, _ in
                    dst.update(from: src.advanced(by: off), count: Self.jointHidden)
                }
            }

            var symbols = 0
            while true {
                let predEmb = await prediction.currentEmbedding()
                var outs = try await joinFused.run(
                    inputs: ["enc_proj_t": encProjT, "pred_emb": predEmb],
                    states: InferenceFunction.MutableViews()
                )
                guard let tokND = outs.remove("token")?.ndArray,
                      let durND = outs.remove("duration")?.ndArray else {
                    throw ASRError.missingOutput("join_fused token/duration")
                }
                var tok: Int32 = 0
                var skip = 0
                tokND.view(as: Int32.self).withUnsafePointer { p, _, _ in tok = p[0] }
                durND.view(as: Int32.self).withUnsafePointer { p, _, _ in skip = Int(p[0]) }

                if tok != blankId {
                    emitted.append(tok)
                    try await prediction.step(token: tok)
                    symbols += 1
                }
                if Self.traceDecode {
                    FileHandle.standardError.write(Data("[tdt/fused] t=\(t) tok=\(tok) skip=\(skip)\n".utf8))
                }
                if skip > 0 {
                    t += skip
                    break
                }
                if tok == blankId || symbols >= Self.maxSymbolsPerStep {
                    t += 1
                    break
                }
            }
        }
        return emitted
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
                    vDSP_maxvi(ptr, 1, &maxV, &maxI, vDSP_Length(tokenLogits))
                    tok = Int32(maxI)
                    var durV: Float = -.infinity
                    for j in 0..<Self.durations.count {
                        let v = ptr[tokenLogits + j]
                        if v > durV { durV = v; durIdx = j }
                    }
                }
                let skip = Self.durations[durIdx]

                if tok != blankId {
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
                if tok == blankId || symbols >= Self.maxSymbolsPerStep {
                    t += 1
                    break
                }
            }
        }
        return emitted
    }
}
