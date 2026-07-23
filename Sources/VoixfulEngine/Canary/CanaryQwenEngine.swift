// Canary-Qwen 2.5B engine — SALM: Conformer encoder + Qwen3-1.7B decoder.
//
// Two modes sharing one asset:
//   ASR pass  (LoRA-merged Qwen3):  mel -> encode_<T> -> [pre ; audio ; post]
//             prompt embeds -> prefill_asr_<L> -> step_asr loop -> transcript.
//   Format pass (pristine Qwen3-1.7B-instruct): the transcript's TOKEN IDS —
//             no host BPE encoder needed — spliced into a formatting prompt
//             -> prefill_llm_<L> -> step_llm loop -> markdown bullets.
//
// The format pass runs when VOIXFUL_CANARY_FORMAT is set (e.g. "bullets" —
// any value enables the convert-time prompt; the CLI's --bullets flag sets
// it). transcribe() then returns the formatted text, so every existing
// surface (file CLI, live mic) gets formatting without protocol changes.
//
// KV caches are engine-owned NDArrays passed as function states to prefill +
// every step (Cohere decode pattern). Generated tokens write their cache rows
// at TRUE positions, progressively overwriting the prefill bucket's padded
// rows; the step graph masks keys > pos, so bucket padding is invisible.
//
// AR decode pays the ~4.5 ms/dispatch floor; transcription is short
// (~1 token/word) and the format pass is bounded by the bullet list length.

import CoreAI
import Foundation

public actor CanaryQwenEngine {
    private let model: AIModel
    private let detokenizer: GraniteBPEDetokenizer   // same tokens.json format
    private let melExtractor: MelExtractor
    private var loaded: [String: InferenceFunction] = [:]
    private var pending: [String: Task<InferenceFunction?, Never>] = [:]

    struct Window {
        let melFrames: Int
        let audioLen: Int      // melFrames / 8
        let asrBucket: Int
        var encodeName: String { "encode_\(melFrames)" }
        var prefillName: String { "prefill_asr_\(asrBucket)" }
    }

    private let windows: [Window]
    private let eosId: Int32
    private let maxSeq: Int
    private let embedN: Int
    private let asrPre: [Int32], asrPost: [Int32]
    private let fmtPre: [Int32], fmtPost: [Int32]
    private let fmtBuckets: [Int]

    // KV caches, shared by prefill + step within a pass (state tensors).
    private var kCache: NDArray
    private var vCache: NDArray

    private static let nMels = 128
    private static let hidden = 2048
    private static let trace = ProcessInfo.processInfo.environment["CANARY_TRACE"] != nil
    private static var formatMode: Bool {
        ProcessInfo.processInfo.environment["VOIXFUL_CANARY_FORMAT"] != nil
    }

    public init(modelURL: URL, piecesURL: URL) async throws {
        _ = piecesURL  // self-contained: canary.json + tokens.json in the asset
        let manifestURL = modelURL.appendingPathComponent("canary.json")
        guard let data = try? Data(contentsOf: manifestURL),
              let m = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              (m["format"] as? Int) == 1,
              let eos = m["eos_id"] as? Int,
              let maxSeq = m["max_seq"] as? Int,
              let embedN = m["embed_n"] as? Int,
              let asrPre = m["asr_pre_ids"] as? [Int],
              let asrPost = m["asr_post_ids"] as? [Int],
              let fmtPre = m["fmt_pre_ids"] as? [Int],
              let fmtPost = m["fmt_post_ids"] as? [Int],
              let fmtBuckets = m["fmt_buckets"] as? [Int],
              let windowList = m["windows"] as? [[String: Any]]
        else {
            throw ASRError.modelLoadFailed("missing/malformed canary.json in \(modelURL.path)")
        }
        self.eosId = Int32(eos)
        self.maxSeq = maxSeq
        self.embedN = embedN
        self.asrPre = asrPre.map(Int32.init)
        self.asrPost = asrPost.map(Int32.init)
        self.fmtPre = fmtPre.map(Int32.init)
        self.fmtPost = fmtPost.map(Int32.init)
        self.fmtBuckets = fmtBuckets.sorted()
        self.windows = windowList.compactMap { w in
            guard let f = w["mel_frames"] as? Int, let a = w["audio_len"] as? Int,
                  let b = w["asr_bucket"] as? Int else { return nil }
            return Window(melFrames: f, audioLen: a, asrBucket: b)
        }.sorted { $0.melFrames < $1.melFrames }
        guard !windows.isEmpty else { throw ASRError.modelLoadFailed("canary.json has no windows") }
        self.detokenizer = try GraniteBPEDetokenizer(
            tokensURL: modelURL.appendingPathComponent("tokens.json"))
        self.melExtractor = MelExtractor(frameStartOffset: (512 - 400) / 2)

        do {
            if let cached = try AIModelCache.default.model(for: modelURL, options: specializationOptions()) {
                self.model = cached
            } else {
                self.model = try await AIModel.specialize(
                    contentsOf: modelURL, options: specializationOptions(),
                    cache: AIModelCache.default, cachePolicy: .persistent
                )
            }
        } catch {
            throw ASRError.modelLoadFailed("specialize(\(modelURL.path)): \(error)")
        }
        // 28 layers, 8 KV heads, head_dim 128 — [L,1,KV,maxSeq,D] fp32.
        self.kCache = NDArray(shape: [28, 1, 8, maxSeq, 128], scalarType: .float32)
        self.vCache = NDArray(shape: [28, 1, 8, maxSeq, 128], scalarType: .float32)
        self.kCache.zeroFloat32()
        self.vCache.zeroFloat32()
    }

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

    public func prewarm() {
        guard ProcessInfo.processInfo.environment["VOIXFUL_PREWARM"] != "0" else { return }
        guard let smallest = windows.first else { return }
        var names = [smallest.encodeName, smallest.prefillName, "step_asr", "embed_\(embedN)"]
        if Self.formatMode, let fb = fmtBuckets.first {
            names += ["prefill_llm_\(fb)", "step_llm"]
        }
        let model = self.model
        for name in names where loaded[name] == nil && pending[name] == nil {
            pending[name] = Task.detached(priority: .utility) {
                (try? model.loadFunction(named: name)) ?? nil
            }
        }
    }

    public func transcribe(audio: [Float]) async throws -> Transcript {
        let t0 = CFAbsoluteTimeGetCurrent()
        let melFlat = await melExtractor.extract(samples: audio)
        let realT = melFlat.count / Self.nMels
        let window = windows.first(where: { $0.melFrames >= realT }) ?? windows[windows.count - 1]
        let useT = min(realT, window.melFrames)
        // Canary's NeMo preprocessor has `normalize: per_feature` — same as
        // Parakeet's. Skipping this leaves the encoder embeds near-orthogonal
        // to reference (measured cos 0.08).
        let normFlat = perFeatureNormalize(melFlat, nMels: Self.nMels, frames: realT)

        var melND = NDArray(shape: [1, Self.nMels, window.melFrames], scalarType: .float32)
        var melMV = melND.mutableView(as: Float.self)
        melMV.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: 0, count: Self.nMels * window.melFrames)
            for m in 0..<Self.nMels {
                let src = m * realT, dst = m * window.melFrames
                for t in 0..<useT { ptr[dst + t] = normFlat[src + t] }
            }
        }
        var lenND = NDArray(shape: [1], scalarType: .int32)
        lenND.setInt32(at: 0, to: Int32(useT))
        let tMel = CFAbsoluteTimeGetCurrent()

        let encode = try await fn(window.encodeName)
        var encOut = try await encode.run(
            inputs: ["mel": melND, "mel_length": lenND],
            states: InferenceFunction.MutableViews())
        guard let audioEmbeds = encOut.remove("audio_embeds")?.ndArray else {
            throw ASRError.missingOutput("audio_embeds")
        }
        let aReal = (useT + 7) / 8
        let tEnc = CFAbsoluteTimeGetCurrent()

        // ASR pass.
        let preE = try await embedIDs(asrPre)
        let postE = try await embedIDs(asrPost)
        let transcriptIds = try await decode(
            prefillName: window.prefillName, stepName: "step_asr",
            bucket: window.asrBucket,
            segments: [.host(preE), .device(audioEmbeds, rows: aReal), .host(postE)],
            maxNew: 256)
        let tASR = CFAbsoluteTimeGetCurrent()

        var text = detokenizer.decode(transcriptIds)
        var tFmt = tASR
        if Self.formatMode, !transcriptIds.isEmpty {
            // Append a pre-closed empty think block ("<think>\n\n</think>\n\n",
            // Qwen3's canonical no-think assistant prefix) so generation starts
            // at the answer. The /no_think soft switch alone is a sampled
            // behavior and flips under palette4/fp16 noise (measured: 384
            // tokens of reasoning instead of bullets).
            let noThink: [Int32] = [151667, 271, 151668, 271]
            let fpreE = try await embedIDs(fmtPre)
            let trE = try await embedIDs(transcriptIds)
            let fpostE = try await embedIDs(fmtPost + noThink)
            let need = fmtPre.count + transcriptIds.count + fmtPost.count + noThink.count
            if let bucket = fmtBuckets.first(where: { $0 >= need }) {
                let fmtIds = try await decode(
                    prefillName: "prefill_llm_\(bucket)", stepName: "step_llm",
                    bucket: bucket,
                    segments: [.host(fpreE), .host(trE), .host(fpostE)],
                    maxNew: 384)
                let formatted = Self.stripThink(detokenizer.decode(fmtIds))
                if !formatted.isEmpty { text = formatted }
            }
            tFmt = CFAbsoluteTimeGetCurrent()
        }

        if Self.trace {
            let line = String(
                format: "[canary/timing] audio=%.1fs melT=%d (window=%d) | mel=%.0fms "
                    + "encode=%.0fms asr=%.0fms (%d tok) fmt=%.0fms\n",
                Double(audio.count) / 16_000, realT, window.melFrames,
                (tMel - t0) * 1000, (tEnc - tMel) * 1000, (tASR - tEnc) * 1000,
                transcriptIds.count, (tFmt - tASR) * 1000)
            FileHandle.standardError.write(Data(line.utf8))
        }
        return Transcript(text: text.trimmingCharacters(in: .whitespacesAndNewlines),
                          segments: [], language: nil)
    }

    // MARK: - Decode plumbing

    private enum Segment {
        case host([Float])                    // flat [n, hidden] floats
        case device(NDArray, rows: Int)       // embeds NDArray, first `rows` rows
    }

    /// Embed token ids via embed_<N> (padded, sliced) -> flat [n*hidden] floats.
    private func embedIDs(_ ids: [Int32]) async throws -> [Float] {
        guard !ids.isEmpty else { return [] }
        precondition(ids.count <= embedN, "id run \(ids.count) > embed bucket \(embedN)")
        var idsND = NDArray(shape: [1, embedN], scalarType: .int32)
        var mv = idsND.mutableView(as: Int32.self)
        mv.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: 0, count: embedN)
            for (i, t) in ids.enumerated() { ptr[i] = t }
        }
        let embed = try await fn("embed_\(embedN)")
        var out = try await embed.run(inputs: ["ids": idsND], states: InferenceFunction.MutableViews())
        guard let e = out.remove("embeds")?.ndArray else { throw ASRError.missingOutput("embeds") }
        var flat = [Float](repeating: 0, count: ids.count * Self.hidden)
        e.view(as: Float.self).withUnsafePointer { src, _, _ in
            flat.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: src, count: ids.count * Self.hidden)
            }
        }
        return flat
    }

    /// Assemble the prompt embeds, prefill, and greedily step until EOS.
    private func decode(
        prefillName: String, stepName: String, bucket: Int,
        segments: [Segment], maxNew: Int
    ) async throws -> [Int32] {
        var embedsND = NDArray(shape: [1, bucket, Self.hidden], scalarType: .float32)
        var promptLen = 0
        var mv = embedsND.mutableView(as: Float.self)
        mv.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: 0, count: bucket * Self.hidden)
            for seg in segments {
                switch seg {
                case .host(let flat):
                    let n = flat.count / Self.hidden
                    flat.withUnsafeBufferPointer { src in
                        ptr.advanced(by: promptLen * Self.hidden)
                            .update(from: src.baseAddress!, count: n * Self.hidden)
                    }
                    promptLen += n
                case .device(let nd, let rows):
                    nd.view(as: Float.self).withUnsafePointer { src, _, _ in
                        ptr.advanced(by: promptLen * Self.hidden)
                            .update(from: src, count: rows * Self.hidden)
                    }
                    promptLen += rows
                }
            }
        }
        precondition(promptLen <= bucket, "prompt \(promptLen) > bucket \(bucket)")

        // CANARY_DUMP=<dir>: write the assembled prompt embeds + prompt len
        // for offline python-runtime replay (debugging aid).
        if let dir = ProcessInfo.processInfo.environment["CANARY_DUMP"] {
            embedsND.view(as: Float.self).withUnsafePointer { p, _, _ in
                let data = Data(bytes: p, count: bucket * Self.hidden * 4)
                try? data.write(to: URL(fileURLWithPath: "\(dir)/swift_embeds_\(prefillName).bin"))
            }
            try? "\(promptLen)".write(
                toFile: "\(dir)/swift_promptlen_\(prefillName).txt", atomically: true, encoding: .utf8)
        }

        var lastIdx = NDArray(shape: [1], scalarType: .int32)
        lastIdx.setInt32(at: 0, to: Int32(promptLen - 1))

        // Prefill is FUNCTIONAL (caches as plain outputs): in-graph state
        // writes corrupt concurrent compute in deep graphs (measured — see
        // convert/canary-qwen bisect). The host loads the returned caches
        // into the persistent state buffers once per pass; the step function
        // then updates them statefully (one-hot write, Cohere-proven).
        let prefill = try await fn(prefillName)
        var out = try await prefill.run(
            inputs: ["embeds": embedsND, "last_idx": lastIdx],
            states: InferenceFunction.MutableViews())
        guard let tok0 = out.remove("token")?.ndArray,
              let kOut = out.remove("k_out")?.ndArray,
              let vOut = out.remove("v_out")?.ndArray else {
            throw ASRError.missingOutput("prefill outputs")
        }
        var tok = tok0.view(as: Int32.self).withUnsafePointer { p, _, _ in p[0] }
        loadPrefillCaches(kOut: kOut, vOut: vOut, rows: bucket)

        let step = try await fn(stepName)
        var ids: [Int32] = []
        var tokND = NDArray(shape: [1], scalarType: .int32)
        var posND = NDArray(shape: [1], scalarType: .int32)
        var pos = promptLen
        while ids.count < maxNew, tok != eosId, pos < maxSeq {
            ids.append(tok)
            tokND.setInt32(at: 0, to: tok)
            posND.setInt32(at: 0, to: Int32(pos))
            var st = InferenceFunction.MutableViews()
            st.insert(&kCache, for: "k_cache")
            st.insert(&vCache, for: "v_cache")
            var sOut = try await step.run(inputs: ["tok": tokND, "pos": posND], states: st)
            guard let nxt = sOut.remove("token")?.ndArray else {
                throw ASRError.missingOutput("token")
            }
            tok = nxt.view(as: Int32.self).withUnsafePointer { p, _, _ in p[0] }
            pos += 1
        }
        return ids
    }

    /// Copy the functional prefill's caches [28, 8, rows, 128] into the
    /// persistent state buffers [28, 1, 8, maxSeq, 128] at rows [0, rows).
    private func loadPrefillCaches(kOut: NDArray, vOut: NDArray, rows: Int) {
        let d = 128, kv = 8, layers = 28
        let maxSeq = self.maxSeq
        // Honor the source view's STRIDES — runtime outputs of this size come
        // back with padded row pitches, so a dense flat copy scrambles rows
        // (measured: step 1 fine, step 2+ garbage).
        func copy(_ src: NDArray, into dst: inout NDArray) {
            src.view(as: Float.self).withUnsafePointer { s, _, strides in
                let sl = strides[0], sh = strides[1], sr = strides[2], sd = strides[3]
                var mv = dst.mutableView(as: Float.self)
                mv.withUnsafeMutablePointer { p, _, _ in
                    for l in 0..<layers {
                        for h in 0..<kv {
                            let dstBase = ((l * kv + h) * maxSeq) * d
                            if sd == 1, sr == d {
                                p.advanced(by: dstBase)
                                    .update(from: s.advanced(by: l * sl + h * sh), count: rows * d)
                            } else if sd == 1 {
                                for r in 0..<rows {
                                    p.advanced(by: dstBase + r * d)
                                        .update(from: s.advanced(by: l * sl + h * sh + r * sr), count: d)
                                }
                            } else {
                                for r in 0..<rows {
                                    let srcBase = l * sl + h * sh + r * sr
                                    for e in 0..<d {
                                        p[dstBase + r * d + e] = s[srcBase + e * sd]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        copy(kOut, into: &kCache)
        copy(vOut, into: &vCache)
    }

    /// Drop a leading Qwen3 `<think>...</think>` block (empty under /no_think).
    static func stripThink(_ s: String) -> String {
        guard let open = s.range(of: "<think>"),
              let close = s.range(of: "</think>") else {
            return s.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        var t = s
        t.removeSubrange(open.lowerBound..<close.upperBound)
        return t.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
