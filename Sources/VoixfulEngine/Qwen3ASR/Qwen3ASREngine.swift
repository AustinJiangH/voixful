// Qwen3-ASR engine — Alibaba/Qwen SALM: audio encoder + Qwen3 LLM decoder.
//
// Pure ASR (transcription only): mel -> encode_<T> -> [pre ; audio ; post]
// prompt embeds -> prefill_asr_<L> -> step_qwen3 loop -> host detokenize.
// Unlike the Canary-Qwen backend (its architectural twin) there is NO second
// formatting pass — Qwen3-ASR emits the transcript directly.
//
// One engine serves BOTH converted checkpoints (qwen3-asr-0.6b and
// qwen3-asr-1.7b): the decoder geometry (hidden size, layer/KV-head/head-dim
// counts, mel bins) is read from `qwen3_asr.json` rather than hardcoded, so the
// same runtime loads either size — they differ only in the asset.
//
// KV caches are engine-owned NDArrays passed as function states to prefill +
// every step (the Cohere/Canary decode pattern): the functional prefill returns
// caches as plain outputs, the host loads them into the persistent state buffers
// once per pass, and step_qwen3 updates them statefully (one-hot write).
//
// The `convert/qwen3-asr` pipeline emits the entrypoints this engine calls —
// encode_<T>, embed_<N>, prefill_asr_<L>, step_qwen3 — plus a `qwen3_asr.json`
// manifest and bundled `tokens.json`. Detection keys this backend on the
// distinctive `step_qwen3` function name.
//
// Two front-end specifics, unlike Canary: (1) the mel is Whisper's log-mel
// (WhisperFeatureExtractor: 128 mel, n_fft 400, hop 160, fixed 30 s window), NOT
// the NeMo per-feature mel — Qwen3-ASR's audio tower was trained on it; (2) the
// decoder emits a `language <Lang><asr_text><transcript>` preamble, so the host
// drops every token up to and including `asr_text_id` before detokenizing.

import CoreAI
import Foundation

public actor Qwen3ASREngine {
    private let model: AIModel                        // audio-encoder instance
    private let decodeModel: AIModel                  // AR-decoder instance (see init)
    private let detokenizer: GraniteBPEDetokenizer   // same tokens.json format
    private let mel: WhisperMel                       // Whisper log-mel front-end
    private var loaded: [String: InferenceFunction] = [:]
    private var pending: [String: Task<InferenceFunction?, Never>] = [:]

    struct Window {
        let melFrames: Int
        let audioLen: Int      // encoder output rows for melFrames — (melFrames/100)*13,
                               // read straight from the manifest (not Canary's /8)
        let asrBucket: Int
        var encodeName: String { "encode_\(melFrames)" }
        var prefillName: String { "prefill_asr_\(asrBucket)" }
    }

    private let windows: [Window]
    private let eosId: Int32
    /// The `<asr_text>` marker id; the decoder prefixes the transcript with a
    /// `language <Lang><asr_text>` preamble, dropped up to and including this id.
    private let asrTextId: Int32?
    private let maxSeq: Int
    private let embedN: Int
    private let asrPre: [Int32], asrPost: [Int32]

    // Decoder geometry — read from the manifest so one engine loads either the
    // 0.6B or the 1.7B checkpoint.
    private let hidden: Int
    private let nMels: Int
    private let numLayers: Int
    private let kvHeads: Int
    private let headDim: Int

    /// ASR decode cap — the loop stops here whether or not EOS was reached, so a
    /// chunk transcribing to more tokens loses its tail at the decoder. Sized for
    /// a dense 30 s window (bucket is 448); the prompt+transcript stays under maxSeq.
    private static let maxDecodeTokens = 400

    /// Largest single-pass audio — the encoder's fixed 30 s window (mel frames ×
    /// 160 samples/frame) bounds it; the decoder budget is looser. Long-form
    /// callers segment below this.
    public nonisolated var maxAudioSamples: Int {
        let encoderWindow = (windows.map(\.melFrames).max() ?? 3000) * 160
        let decoderBudget = (Self.maxDecodeTokens / 7) * 16_000
        return min(encoderWindow, decoderBudget)
    }

    // KV caches, shared by prefill + step within a pass (state tensors).
    private var kCache: NDArray
    private var vCache: NDArray
    /// Element type of the KV state I/O the loaded decode graph expects —
    /// `.float16` (fast path, decoder can land on the ANE) or `.float32`
    /// (legacy). Detected once at load; drives cache allocation and the
    /// prefill-cache copy.
    private let kvScalarType: NDArray.ScalarType

    private static let trace = ProcessInfo.processInfo.environment["QWEN3_TRACE"] != nil

    public init(modelURL: URL, piecesURL: URL) async throws {
        _ = piecesURL  // self-contained: qwen3_asr.json + tokens.json in the asset
        let manifestURL = modelURL.appendingPathComponent("qwen3_asr.json")
        guard let data = try? Data(contentsOf: manifestURL),
              let m = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              (m["format"] as? Int) == 1,
              let eos = m["eos_id"] as? Int,
              let maxSeq = m["max_seq"] as? Int,
              let embedN = m["embed_n"] as? Int,
              let hidden = m["hidden"] as? Int,
              let numLayers = m["num_layers"] as? Int,
              let kvHeads = m["kv_heads"] as? Int,
              let headDim = m["head_dim"] as? Int,
              let asrPre = m["asr_pre_ids"] as? [Int],
              let asrPost = m["asr_post_ids"] as? [Int],
              let windowList = m["windows"] as? [[String: Any]]
        else {
            throw ASRError.modelLoadFailed("missing/malformed qwen3_asr.json in \(modelURL.path)")
        }
        self.eosId = Int32(eos)
        self.asrTextId = (m["asr_text_id"] as? Int).map(Int32.init)
        self.maxSeq = maxSeq
        self.embedN = embedN
        self.hidden = hidden
        self.numLayers = numLayers
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.nMels = (m["n_mels"] as? Int) ?? 128
        self.asrPre = asrPre.map(Int32.init)
        self.asrPost = asrPost.map(Int32.init)
        self.windows = windowList.compactMap { w in
            guard let f = w["mel_frames"] as? Int, let a = w["audio_len"] as? Int,
                  let b = w["asr_bucket"] as? Int else { return nil }
            return Window(melFrames: f, audioLen: a, asrBucket: b)
        }.sorted { $0.melFrames < $1.melFrames }
        guard !windows.isEmpty else { throw ASRError.modelLoadFailed("qwen3_asr.json has no windows") }
        self.detokenizer = try GraniteBPEDetokenizer(
            tokensURL: modelURL.appendingPathComponent("tokens.json"))
        // Qwen3-ASR's audio tower was trained on Whisper's log-mel front-end
        // (WhisperFeatureExtractor): 128 mel, n_fft 400, hop 160. Size the fixed
        // window to the LARGEST converted encoder window (mel frames × hop) so the
        // mel output and the encoder graph always agree in frame count — otherwise
        // a manifest window ≠ 3000 would silently drop audio past the mel window.
        let melHop = 160
        let maxMelFrames = windows.map(\.melFrames).max() ?? 3000
        self.mel = WhisperMel(
            sampleRate: 16_000, nMels: self.nMels,
            nFFT: 400, hop: melHop, chunkSamples: maxMelFrames * melHop)

        // Two specialized instances of the SAME asset so the audio encoder and
        // the AR decoder can land on DIFFERENT compute units — the sanctioned
        // per-function-placement workaround (Core AI has no per-function options
        // API; same split Cohere uses). Measured on this asset: the audio tower
        // is faster AND accurate on CPU, whereas ANE's fp16 compute flips argmax
        // on its embeds (→ a wrong-but-plausible transcript); the AR decoder is
        // ~3× faster on ANE and stays correct when fed the CPU-computed embeds.
        // So the default is encode→CPU, decode→ANE (fast + correct); the split
        // coalesces to one instance when both resolve to the same unit.
        let encKind = Self.encodeComputeKind, decKind = Self.decodeComputeKind
        let encModel: AIModel
        let decModel: AIModel
        do {
            if encKind == decKind {
                encModel = try await Self.specialize(modelURL, Self.options(encKind))
                decModel = encModel
            } else {
                // Independent specializations of the same asset — run concurrently
                // so cold-start is max(enc, dec), not enc + dec (each is multi-second
                // on the 1.7B).
                async let enc = Self.specialize(modelURL, Self.options(encKind))
                async let dec = Self.specialize(modelURL, Self.options(decKind))
                encModel = try await enc
                decModel = try await dec
            }
        } catch {
            throw ASRError.modelLoadFailed("specialize(\(modelURL.path)): \(error)")
        }
        self.model = encModel
        self.decodeModel = decModel
        // KV state buffers [L, 1, KV, maxSeq, D]. Element type is whatever the
        // decode graph's KV state declares: fp16 for new assets (fp16 state I/O
        // lets the AR decoder land on the ANE), fp32 for legacy. Preferred
        // detection is the graph's own state descriptor; the manifest "kv_dtype"
        // field is a fallback (fp32 default). Read from the DECODE instance —
        // that's where step_qwen3 runs.
        let decodeInstance = decModel
        var graphKV: NDArray.ScalarType? = nil
        if let desc = decodeInstance.functionDescriptor(for: "step_qwen3"),
           case .ndArray(let nd)? = desc.stateDescriptor(of: "k_cache") {
            graphKV = nd.scalarType
        }
        self.kvScalarType = Self.kvScalarType(
            graphDtype: graphKV, manifestKVDtype: m["kv_dtype"] as? String)
        self.kCache = NDArray(shape: [numLayers, 1, kvHeads, maxSeq, headDim], scalarType: self.kvScalarType)
        self.vCache = NDArray(shape: [numLayers, 1, kvHeads, maxSeq, headDim], scalarType: self.kvScalarType)
        if self.kvScalarType == .float16 {
            self.kCache.zeroFloat16()
            self.vCache.zeroFloat16()
        } else {
            self.kCache.zeroFloat32()
            self.vCache.zeroFloat32()
        }
    }

    /// Resolve the KV cache element type. Preferred source is the loaded decode
    /// graph's own state descriptor (`graphDtype`); falls back to the manifest
    /// "kv_dtype" field, then to fp32 (legacy). Pure/deterministic — unit-testable
    /// without an asset.
    static func kvScalarType(
        graphDtype: NDArray.ScalarType?, manifestKVDtype: String?
    ) -> NDArray.ScalarType {
        if let g = graphDtype { return g == .float16 ? .float16 : .float32 }
        if manifestKVDtype == "float16" { return .float16 }
        return .float32
    }

    /// Compute-unit selection as a small comparable token so the two instances
    /// coalesce when they resolve to the same unit.
    private enum ComputeKind: Equatable { case cpu, gpu, ane, auto }

    private static func kind(from env: String?) -> ComputeKind? {
        switch env?.lowercased() {
        case "cpu": return .cpu
        case "gpu": return .gpu
        case "ane", "neuralengine", "neural-engine": return .ane
        case "auto", "default": return .auto
        default: return nil
        }
    }

    /// Encode-instance placement — CPU by default (the audio tower is fastest AND
    /// accurate there; ANE fp16 flips argmax on its embeds). `VOIXFUL_COMPUTE`
    /// overrides.
    private static var encodeComputeKind: ComputeKind {
        kind(from: ProcessInfo.processInfo.environment["VOIXFUL_COMPUTE"]) ?? .cpu
    }
    /// Decode-instance placement — ANE by default (~3× faster and correct on the
    /// CPU-computed embeds). `VOIXFUL_COMPUTE_DECODE` overrides; otherwise it
    /// follows `VOIXFUL_COMPUTE` when that is set (so pinning the whole model to
    /// one unit still works), else ANE.
    private static var decodeComputeKind: ComputeKind {
        if let d = kind(from: ProcessInfo.processInfo.environment["VOIXFUL_COMPUTE_DECODE"]) { return d }
        if let c = kind(from: ProcessInfo.processInfo.environment["VOIXFUL_COMPUTE"]) { return c }
        return .ane
    }

    private static func options(_ k: ComputeKind) -> SpecializationOptions {
        switch k {
        case .cpu: return SpecializationOptions(preferredComputeUnitKind: .cpu)
        case .gpu: return SpecializationOptions(preferredComputeUnitKind: .gpu)
        case .ane: return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        case .auto: return .default
        }
    }

    /// Specialize the asset with `opts`, reusing the process cache. Static so the
    /// initializer can call it before `self` is fully initialized.
    private static func specialize(_ url: URL, _ opts: SpecializationOptions) async throws -> AIModel {
        if let cached = try AIModelCache.default.model(for: url, options: opts) {
            return cached
        }
        return try await AIModel.specialize(
            contentsOf: url, options: opts,
            cache: AIModelCache.default, cachePolicy: .persistent)
    }

    /// The encoder runs on the encode instance; embed/prefill/step on the decode
    /// instance. When the two coalesce, both point at the same AIModel.
    private func modelFor(_ name: String) -> AIModel {
        name.hasPrefix("encode_") ? model : decodeModel
    }

    private func fn(_ name: String) async throws -> InferenceFunction {
        if let f = loaded[name] { return f }
        if let t = pending.removeValue(forKey: name) {
            if let f = await t.value {
                loaded[name] = f
                return f
            }
        }
        guard let f = try modelFor(name).loadFunction(named: name) else {
            throw ASRError.missingFunction(name)
        }
        loaded[name] = f
        return f
    }

    public func prewarm() {
        guard ProcessInfo.processInfo.environment["VOIXFUL_PREWARM"] != "0" else { return }
        guard let smallest = windows.first else { return }
        let names = [smallest.encodeName, smallest.prefillName, "step_qwen3", "embed_\(embedN)"]
        let encModel = self.model, decModel = self.decodeModel
        for name in names where loaded[name] == nil && pending[name] == nil {
            let target = name.hasPrefix("encode_") ? encModel : decModel
            pending[name] = Task.detached(priority: .utility) {
                (try? target.loadFunction(named: name)) ?? nil
            }
        }
    }

    public func transcribe(audio: [Float]) async throws -> Transcript {
        let t0 = CFAbsoluteTimeGetCurrent()
        // Whisper log-mel is a FIXED [nMels × 3000] (pad/trim to a 30 s window),
        // so there is a single encoder window; match it by frame count.
        let melFlat = mel.logMel(samples: audio)
        let frames = mel.melFrames
        let window = windows.first(where: { $0.melFrames == frames }) ?? windows[windows.count - 1]
        let nMels = self.nMels

        var melND = NDArray(shape: [1, nMels, window.melFrames], scalarType: .float32)
        var melMV = melND.mutableView(as: Float.self)
        melMV.withUnsafeMutablePointer { ptr, _, _ in
            if frames == window.melFrames {
                // Common path: WhisperMel emits exactly [nMels × window.melFrames]
                // (window sized to the mel in init) — one bulk copy, no zero-fill.
                melFlat.withUnsafeBufferPointer { src in
                    ptr.update(from: src.baseAddress!, count: nMels * window.melFrames)
                }
            } else {
                // Defensive: mel/window frame counts differ — zero-pad per row.
                ptr.update(repeating: 0, count: nMels * window.melFrames)
                let copyF = min(frames, window.melFrames)
                for m in 0..<nMels {
                    let src = m * frames, dst = m * window.melFrames
                    for t in 0..<copyF { ptr[dst + t] = melFlat[src + t] }
                }
            }
        }
        var lenND = NDArray(shape: [1], scalarType: .int32)
        lenND.setInt32(at: 0, to: Int32(window.melFrames))
        let tMel = CFAbsoluteTimeGetCurrent()

        let encode = try await fn(window.encodeName)
        var encOut = try await encode.run(
            inputs: ["mel": melND, "mel_length": lenND],
            states: InferenceFunction.MutableViews())
        guard let audioEmbeds = encOut.remove("audio_embeds")?.ndArray else {
            throw ASRError.missingOutput("audio_embeds")
        }
        // The encoder emits a full 30 s window of audio tokens; the exporter's
        // validated path splices ALL of them (silence-padded tail included), so
        // use the manifest audio_len — not a duration-scaled slice.
        let audioRows = window.audioLen
        let tEnc = CFAbsoluteTimeGetCurrent()

        let preE = try await embedIDs(asrPre)
        let postE = try await embedIDs(asrPost)
        let transcriptIds = try await decode(
            prefillName: window.prefillName, stepName: "step_qwen3",
            bucket: window.asrBucket,
            segments: [.host(preE), .device(audioEmbeds, rows: audioRows), .host(postE)],
            maxNew: Self.maxDecodeTokens)
        let tASR = CFAbsoluteTimeGetCurrent()

        // Drop the `language <Lang><asr_text>` preamble the decoder emits: keep
        // only tokens after the asr_text marker (the marker id is blanked in
        // tokens.json, but the language words ahead of it are ordinary tokens).
        let body: [Int32]
        if let marker = asrTextId, let idx = transcriptIds.firstIndex(of: marker) {
            body = Array(transcriptIds[(idx + 1)...])
        } else {
            body = transcriptIds
        }
        let text = detokenizer.decode(body)

        if Self.trace {
            let line = String(
                format: "[qwen3/timing] audio=%.1fs window=%d | mel=%.0fms "
                    + "encode=%.0fms asr=%.0fms (%d tok)\n",
                Double(audio.count) / 16_000, window.melFrames,
                (tMel - t0) * 1000, (tEnc - tMel) * 1000, (tASR - tEnc) * 1000,
                transcriptIds.count)
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
        let hidden = self.hidden
        var idsND = NDArray(shape: [1, embedN], scalarType: .int32)
        var mv = idsND.mutableView(as: Int32.self)
        mv.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: 0, count: embedN)
            for (i, t) in ids.enumerated() { ptr[i] = t }
        }
        let embed = try await fn("embed_\(embedN)")
        var out = try await embed.run(inputs: ["ids": idsND], states: InferenceFunction.MutableViews())
        guard let e = out.remove("embeds")?.ndArray else { throw ASRError.missingOutput("embeds") }
        var flat = [Float](repeating: 0, count: ids.count * hidden)
        // Honor the view's strides — runtime outputs of this size can come back
        // with a padded row pitch (row stride > hidden), same as the prefill
        // caches; a flat copy would misread every row after the first.
        e.view(as: Float.self).withUnsafePointer { src, _, strides in
            let rowStride = strides[1], elemStride = strides[2]
            flat.withUnsafeMutableBufferPointer { dst in
                if elemStride == 1 {
                    for r in 0..<ids.count {
                        dst.baseAddress!.advanced(by: r * hidden)
                            .update(from: src.advanced(by: r * rowStride), count: hidden)
                    }
                } else {
                    for r in 0..<ids.count {
                        for c in 0..<hidden { dst[r * hidden + c] = src[r * rowStride + c * elemStride] }
                    }
                }
            }
        }
        return flat
    }

    /// Assemble the prompt embeds, prefill, and greedily step until EOS.
    private func decode(
        prefillName: String, stepName: String, bucket: Int,
        segments: [Segment], maxNew: Int
    ) async throws -> [Int32] {
        let hidden = self.hidden
        var embedsND = NDArray(shape: [1, bucket, hidden], scalarType: .float32)
        var promptLen = 0
        var mv = embedsND.mutableView(as: Float.self)
        mv.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: 0, count: bucket * hidden)
            for seg in segments {
                switch seg {
                case .host(let flat):
                    let n = flat.count / hidden
                    flat.withUnsafeBufferPointer { src in
                        ptr.advanced(by: promptLen * hidden)
                            .update(from: src.baseAddress!, count: n * hidden)
                    }
                    promptLen += n
                case .device(let nd, let rows):
                    // Honor strides: the encoder's audio_embeds can return with a
                    // padded row pitch, so copy row-by-row at the view's stride
                    // rather than as one flat block (which would splice garbage).
                    nd.view(as: Float.self).withUnsafePointer { src, _, strides in
                        let rowStride = strides[1], elemStride = strides[2]
                        let base = promptLen * hidden
                        if elemStride == 1 {
                            for r in 0..<rows {
                                ptr.advanced(by: base + r * hidden)
                                    .update(from: src.advanced(by: r * rowStride), count: hidden)
                            }
                        } else {
                            for r in 0..<rows {
                                for c in 0..<hidden { ptr[base + r * hidden + c] = src[r * rowStride + c * elemStride] }
                            }
                        }
                    }
                    promptLen += rows
                }
            }
        }
        precondition(promptLen <= bucket, "prompt \(promptLen) > bucket \(bucket)")

        var lastIdx = NDArray(shape: [1], scalarType: .int32)
        lastIdx.setInt32(at: 0, to: Int32(promptLen - 1))

        // Prefill is FUNCTIONAL (caches as plain outputs): the host loads the
        // returned caches into the persistent state buffers once per pass; the
        // step function then updates them statefully (one-hot write).
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
            // Repetition guard: greedy AR decode can loop on non-speech input,
            // emitting a short cycle until the maxNew cap. Trip only on a LONG
            // sustained cycle (32 tokens, period ≤3) — long enough that genuine
            // repeated speech (a digit string, "yeah yeah yeah", a repeated CJK
            // character) survives, while a true decoder loop is still caught well
            // before maxNew. maxDecodeTokens is the hard backstop either way.
            let repeatWindow = 32
            if ids.count >= repeatWindow {
                let tail = Array(ids.suffix(repeatWindow))
                for period in 1...3 {
                    if (period..<repeatWindow).allSatisfy({ tail[$0] == tail[$0 - period] }) {
                        return ids
                    }
                }
            }
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

    /// Copy the functional prefill's caches [L, KV, rows, D] into the persistent
    /// state buffers [L, 1, KV, maxSeq, D] at rows [0, rows). The prefill's
    /// k_out/v_out come back in the same element type as the KV state.
    private func loadPrefillCaches(kOut: NDArray, vOut: NDArray, rows: Int) {
        if kvScalarType == .float16 {
            copyPrefillCache(Float16.self, kOut, into: &kCache, rows: rows)
            copyPrefillCache(Float16.self, vOut, into: &vCache, rows: rows)
        } else {
            copyPrefillCache(Float.self, kOut, into: &kCache, rows: rows)
            copyPrefillCache(Float.self, vOut, into: &vCache, rows: rows)
        }
    }

    /// Strided element copy of one prefill cache into its persistent buffer,
    /// generic over the element type so the same logic serves both the fp16 and
    /// the legacy fp32 KV layouts (`Element` == `Float16` or `Float`).
    private func copyPrefillCache<Element: BitwiseCopyable>(
        _: Element.Type, _ src: NDArray, into dst: inout NDArray, rows: Int
    ) {
        let d = headDim, kv = kvHeads, layers = numLayers
        let maxSeq = self.maxSeq
        // Honor the source view's STRIDES — runtime outputs of this size come
        // back with padded row pitches. Strides are in ELEMENTS, so the
        // arithmetic is identical for fp16 and fp32.
        src.view(as: Element.self).withUnsafePointer { s, _, strides in
            let sl = strides[0], sh = strides[1], sr = strides[2], sd = strides[3]
            var mv = dst.mutableView(as: Element.self)
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
}
