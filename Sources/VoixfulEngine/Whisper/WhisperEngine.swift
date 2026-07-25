// Whisper — attention encoder-decoder (AED) engine. Same shape as Cohere
// Transcribe (full-context Conformer/Transformer encoder feeding an
// autoregressive cross-attention decoder), so the runtime structure mirrors
// `CohereTranscribeEngine`. The Whisper-specific differences:
//
//   - FIXED 30 s input: Whisper always pads/truncates audio to 30 s -> 3000
//     log-mel frames -> the encoder downsamples 2x to 1500 frames. So the
//     single-pass window is inherently 30 s; `maxAudioSamples` = 30·16000.
//     (Long-form is handled by the shared LongForm segmenter.)
//   - Whisper's own log-mel front-end (see `WhisperMel`): 80 mel channels for
//     tiny/base/small/medium/large-v1/v2, 128 for large-v3; log10 + global
//     (x+4)/4 normalization, no pre-emphasis.
//   - Byte-level BPE tokenizer (see `WhisperTokenizer`), NOT SentencePiece.
//   - Decoder priming: the decode loop starts from the special-token prefix
//     <|startoftranscript|> <|lang|> <|transcribe|> <|notimestamps|>, then text
//     tokens until <|endoftext|> (EOT).
//
// The asset is self-contained: `whisper.json` (architecture constants + special
// token ids) and `tokens.txt` (byte-level BPE vocab) are bundled inside the
// .aimodel at convert time, so the specialize cache keys on the final contents.
//
//   whisper_encode(input_features[1,nMels,3000]) -> encoder_hidden_states[1,1500,d]
//   whisper_decode(input_id, position_id, encoder_hidden_states,
//                  self_attention_mask[1,1,1,M], k_cache[L,1,H,M,D], v_cache[…])
//       -> logits[1,1,vocab], k_cache_out, v_cache_out
//
// Both functions are stateless — the host owns the self-attention KV cache and
// builds the causal mask (mirroring Cohere's legacy `decode_step`). Whisper
// cross-attention needs no mask: the encoder always sees the full 30 s window,
// so every one of the 1500 encoder frames is valid.

import Accelerate
import CoreAI
import Foundation

public actor WhisperEngine {
    private let model: AIModel
    private let available: Set<String>
    private var loaded: [String: InferenceFunction] = [:]
    private var pending: [String: Task<InferenceFunction?, Never>] = [:]
    private let tokenizer: WhisperTokenizer
    private let mel: WhisperMel
    private let cfg: Config

    /// Architecture constants + special token ids, read from the asset's
    /// `whisper.json` so a model-size swap (base/small/large-v3) needs no code
    /// change. All shapes the graphs were traced against live here.
    struct Config {
        let nMels: Int
        let melFrames: Int      // 3000 (30 s)
        let chunkSamples: Int   // 480000 (30 s @ 16 kHz)
        let sampleRate: Int
        let nFFT: Int
        let hop: Int
        let tEnc: Int           // encoder output frames (1500)
        let dModel: Int
        let nLayers: Int
        let nHeads: Int
        let headDim: Int
        let vocab: Int
        let maxCacheLen: Int    // decoder token cap (self-attention cache slots)
        let sot: Int32          // <|startoftranscript|>
        let eot: Int32          // <|endoftext|>
        let transcribe: Int32   // <|transcribe|>
        let noTimestamps: Int32 // <|notimestamps|>
        let langToken: Int32    // <|en|> (or the requested locale's language)
        let specialStart: Int   // first special-token id (tokenizer skips >= this)
    }

    private static let maskNeg: Float = -1e9
    private static let traceDecode = ProcessInfo.processInfo.environment["WHISPER_TRACE"] != nil

    /// Largest single-pass audio: the fixed 30 s encoder window. Whisper never
    /// transcribes more than 30 s in one pass; LongForm segments beyond that.
    public nonisolated var maxAudioSamples: Int { 30 * 16_000 }

    public init(modelURL: URL, piecesURL: URL) async throws {
        // piecesURL is the FullContextEngine protocol slot; Whisper is
        // self-contained (whisper.json + tokens.txt inside the asset).
        _ = piecesURL
        let cfg = try Self.loadConfig(modelURL: modelURL, locale: Locale(identifier: "en-US"))
        self.cfg = cfg

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

        let names = try ModelBackend.assetFunctions(modelURL: modelURL).names
        guard names.contains("whisper_encode") else { throw ASRError.missingFunction("whisper_encode") }
        guard names.contains("whisper_decode") else { throw ASRError.missingFunction("whisper_decode") }
        self.available = names

        let tokenizer = try WhisperTokenizer(
            tokensURL: modelURL.appendingPathComponent("tokens.txt"),
            specialTokenStart: cfg.specialStart)
        guard tokenizer.count == cfg.vocab else {
            throw ASRError.modelLoadFailed(
                "vocab mismatch: tokens.txt has \(tokenizer.count) pieces, "
                + "expected \(cfg.vocab) for Whisper — wrong asset / tokens.txt?")
        }
        self.tokenizer = tokenizer
        self.mel = WhisperMel(
            sampleRate: cfg.sampleRate, nMels: cfg.nMels, nFFT: cfg.nFFT,
            hop: cfg.hop, chunkSamples: cfg.chunkSamples)
    }

    /// Parse `whisper.json`. The language token defaults to the requested
    /// locale's language (falling back to `<|en|>`), matching Whisper's language
    /// prompt token; the model still auto-detects when primed with a mismatch.
    private static func loadConfig(modelURL: URL, locale: Locale) throws -> Config {
        let url = modelURL.appendingPathComponent("whisper.json")
        guard let data = try? Data(contentsOf: url),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              (obj["format"] as? Int) == 1,
              let nMels = obj["n_mels"] as? Int,
              let melFrames = obj["mel_frames"] as? Int,
              let chunkSamples = obj["chunk_samples"] as? Int,
              let sampleRate = obj["sample_rate"] as? Int,
              let nFFT = obj["n_fft"] as? Int,
              let hop = obj["hop_length"] as? Int,
              let tEnc = obj["n_audio_ctx"] as? Int,
              let dModel = obj["d_model"] as? Int,
              let nLayers = obj["n_layer"] as? Int,
              let nHeads = obj["n_head"] as? Int,
              let vocab = obj["vocab_size"] as? Int,
              let maxCacheLen = obj["max_decode_tokens"] as? Int,
              let sot = obj["sot"] as? Int,
              let eot = obj["eot"] as? Int,
              let transcribe = obj["transcribe"] as? Int,
              let noTimestamps = obj["no_timestamps"] as? Int,
              let specialStart = obj["special_token_start"] as? Int
        else {
            throw ASRError.modelLoadFailed("missing/malformed whisper.json in \(modelURL.path)")
        }
        // Language prompt token: look up the locale's language in the map, else
        // the bundled English default.
        let langs = obj["languages"] as? [String: Int] ?? [:]
        let lang = locale.language.languageCode?.identifier ?? "en"
        let englishDefault = (obj["lang_en"] as? Int) ?? Int(sot) + 1
        var langToken = langs[lang] ?? englishDefault
        // Defensive: a language absent from the checkpoint can resolve to a bogus
        // id that aliases eot (e.g. pre-fix manifests mapped <|yue|> -> eot);
        // priming the decoder with EOT stops it instantly. Fall back to English.
        if langToken <= eot { langToken = englishDefault }
        return Config(
            nMels: nMels, melFrames: melFrames, chunkSamples: chunkSamples,
            sampleRate: sampleRate, nFFT: nFFT, hop: hop, tEnc: tEnc, dModel: dModel,
            nLayers: nLayers, nHeads: nHeads, headDim: dModel / nHeads, vocab: vocab,
            maxCacheLen: maxCacheLen, sot: Int32(sot), eot: Int32(eot),
            transcribe: Int32(transcribe), noTimestamps: Int32(noTimestamps),
            langToken: Int32(langToken), specialStart: specialStart)
    }

    /// Load (and cache) a function by name — lazy so weight materialization is
    /// paid once. Awaits an in-flight prewarm load of the same name.
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

    /// Background-load the encoder + decoder so weight materialization overlaps
    /// audio capture. `VOIXFUL_PREWARM=0` disables.
    public func prewarm() {
        guard ProcessInfo.processInfo.environment["VOIXFUL_PREWARM"] != "0" else { return }
        let model = self.model
        for name in ["whisper_encode", "whisper_decode"]
        where available.contains(name) && loaded[name] == nil && pending[name] == nil {
            pending[name] = Task.detached(priority: .utility) {
                (try? model.loadFunction(named: name)) ?? nil
            }
        }
    }

    public func transcribe(audio: [Float]) async throws -> Transcript {
        let t0 = CFAbsoluteTimeGetCurrent()

        // Mel -> fixed [1, nMels, melFrames] (30 s window, pad/trim in-front-end).
        let melFlat = mel.logMel(samples: audio)  // [nMels × melFrames]
        var melND = NDArray(shape: [1, cfg.nMels, cfg.melFrames], scalarType: .float32)
        var melMV = melND.mutableView(as: Float.self)
        melMV.withUnsafeMutablePointer { ptr, _, _ in
            melFlat.withUnsafeBufferPointer { src in
                ptr.update(from: src.baseAddress!, count: cfg.nMels * cfg.melFrames)
            }
        }
        let tMel = CFAbsoluteTimeGetCurrent()

        // Encode once -> [1, tEnc, dModel].
        let encode = try await fn("whisper_encode")
        var encOut = try await encode.run(
            inputs: ["input_features": melND],
            states: InferenceFunction.MutableViews())
        guard let enc = encOut.remove("encoder_hidden_states")?.ndArray else {
            throw ASRError.missingOutput("encoder_hidden_states")
        }
        let tEnc = CFAbsoluteTimeGetCurrent()

        let emitted = try await decode(step: try await fn("whisper_decode"), enc: enc)
        if Self.traceDecode {
            let tDec = CFAbsoluteTimeGetCurrent()
            let line = String(
                format: "[whisper/timing] audio=%.1fs | mel=%.0fms encode=%.0fms "
                    + "decode=%.0fms (%d tokens)\n",
                Double(audio.count) / 16_000, (tMel - t0) * 1000, (tEnc - tMel) * 1000,
                (tDec - tEnc) * 1000, emitted.count)
            FileHandle.standardError.write(Data(line.utf8))
        }
        return Transcript(text: tokenizer.decode(emitted), segments: [], language: nil)
    }

    /// Greedy autoregressive decode with an external self-attention KV cache.
    /// Primed with the SOT prefix (<|sot|> <|lang|> <|transcribe|>
    /// <|notimestamps|>): those tokens are fed to fill the cache but produce no
    /// text; generation then continues from the last prefix token until EOT.
    private func decode(step decodeFn: InferenceFunction, enc: NDArray) async throws -> [Int32] {
        let cacheShape = [cfg.nLayers, 1, cfg.nHeads, cfg.maxCacheLen, cfg.headDim]
        var kCache = NDArray(shape: cacheShape, scalarType: .float32)
        var vCache = NDArray(shape: cacheShape, scalarType: .float32)
        kCache.zeroFloat32()
        vCache.zeroFloat32()

        var selfMask = NDArray(shape: [1, 1, 1, cfg.maxCacheLen], scalarType: .float32)
        var inputND = NDArray(shape: [1, 1], scalarType: .int32)
        var posND = NDArray(shape: [1, 1], scalarType: .int32)

        // The forced prefix: consumed to prime the cache, never emitted.
        let prefix: [Int32] = [cfg.sot, cfg.langToken, cfg.transcribe, cfg.noTimestamps]

        var emitted: [Int32] = []
        var prev = prefix[0]
        var prefixIdx = 1  // next forced token to feed (prefix[0] already loaded)
        var hitEOS = false
        for pos in 0..<cfg.maxCacheLen {
            inputND.setInt32(at: 0, to: prev)
            posND.setInt32(at: 0, to: Int32(pos))
            // Causal self-attention mask: attend to cache positions 0...pos.
            var selfMV = selfMask.mutableView(as: Float.self)
            selfMV.withUnsafeMutablePointer { ptr, _, _ in
                for j in 0..<cfg.maxCacheLen { ptr[j] = j <= pos ? 0 : Self.maskNeg }
            }

            var out = try await decodeFn.run(
                inputs: [
                    "input_id": inputND, "position_id": posND,
                    "encoder_hidden_states": enc,
                    "self_attention_mask": selfMask,
                    "k_cache": kCache, "v_cache": vCache,
                ],
                states: InferenceFunction.MutableViews()
            )
            guard let logits = out.remove("logits")?.ndArray,
                  let kOut = out.remove("k_cache_out")?.ndArray,
                  let vOut = out.remove("v_cache_out")?.ndArray else {
                throw ASRError.missingOutput("whisper_decode outputs")
            }
            kCache = kOut
            vCache = vOut

            // Still inside the forced prefix: keep feeding it, ignore the logits.
            if prefixIdx < prefix.count {
                prev = prefix[prefixIdx]
                prefixIdx += 1
                continue
            }

            // Greedy over text tokens only; stop when EOT outscores them (never
            // emit a special/timestamp token — we primed <|notimestamps|>).
            let next = argmaxText(logits)
            let eotV = logitValue(logits, at: Int(cfg.eot))
            if Self.traceDecode {
                FileHandle.standardError.write(Data("[whisper] pos=\(pos) tok=\(next)\n".utf8))
            }
            if eotV >= next.value { hitEOS = true; break }
            emitted.append(next.id)
            prev = next.id
        }
        if !hitEOS {
            FileHandle.standardError.write(Data((
                "[whisper] warning: hit the \(cfg.maxCacheLen)-token decode limit without EOT — "
                + "transcript is likely TRUNCATED (30 s chunk exceeds the token budget).\n").utf8))
        }
        return emitted
    }

    /// Argmax over the text/byte token range `[0, specialStart)` — excludes all
    /// special/language/timestamp tokens so greedy decode emits only real text.
    private func argmaxText(_ logits: NDArray) -> (id: Int32, value: Float) {
        var maxV: Float = -.infinity
        var maxI: vDSP_Length = 0
        logits.view(as: Float.self).withUnsafePointer { ptr, _, _ in
            vDSP_maxvi(ptr, 1, &maxV, &maxI, vDSP_Length(cfg.specialStart))
        }
        return (Int32(maxI), maxV)
    }

    private func logitValue(_ logits: NDArray, at index: Int) -> Float {
        logits.view(as: Float.self).withUnsafePointer { ptr, _, _ in ptr[index] }
    }
}
