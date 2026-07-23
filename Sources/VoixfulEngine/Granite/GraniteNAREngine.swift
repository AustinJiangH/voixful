// Granite Speech 4.1 2B NAR engine — full-context, non-autoregressive.
//
// Two dispatches per clip, NO autoregressive loop (the property that makes
// this backend immune to the ~4.5 ms/step Core AI dispatch floor the Cohere
// AED decoder pays):
//
//   encode_<F>(audio [1, F*320])            -> audio_embeds [1, F/5, 2048],
//                                              ctc_ids [1, F/4] i32
//   edit_<A>_<U>(audio_embeds, text_ids, attn_mask) -> ids [1, U] i32
//
// Host glue between them (microseconds, pure Swift):
//   1. CTC-collapse ctc_ids (unique-consecutive, drop blank) -> hypothesis
//   2. interleave insertion slots: [blank, t1, blank, t2, ..., blank]
//      (min length from granite.json), pad to the window's text bucket with
//      blank + a zeroed attention mask (masked keys are invisible to real
//      positions -> bucket padding is parity-exact)
//   3. CTC-collapse the LLM's output ids -> final tokens -> byte-level BPE
//
// The mel front-end is IN-GRAPH (exact torchaudio parity by construction),
// so the engine feeds raw 16 kHz samples. Audio is zero-padded to the
// window — verified transcript-invariant (granite_convert.parity): silence
// is in-distribution and the padded tail CTC-decodes to pure blanks.
//
// Window/bucket table, blank id, and min edit length come from the asset's
// `granite.json`; the id->bytes table from `tokens.json` (both written at
// convert time, so the specialize cache keys on the final asset contents).

import CoreAI
import Foundation

public actor GraniteNAREngine {
    private let model: AIModel
    private let detokenizer: GraniteBPEDetokenizer
    private var loaded: [String: InferenceFunction] = [:]
    private var pending: [String: Task<InferenceFunction?, Never>] = [:]

    struct Window {
        let frames: Int      // encoder frames (50 Hz); samples = frames * 320
        let audioLen: Int    // frames / 5 (10 Hz LLM audio tokens)
        let textBucket: Int  // static text-id slot count of the paired edit fn
        var encodeName: String { "encode_\(frames)" }
        var editName: String { "edit_\(audioLen)_\(textBucket)" }
    }

    private let windows: [Window]  // ascending by frames
    private let blankId: Int32
    private let minEditLength: Int

    private static let samplesPerFrame = 320
    private static let trace = ProcessInfo.processInfo.environment["GRANITE_TRACE"] != nil

    public init(modelURL: URL, piecesURL: URL) async throws {
        // piecesURL is the FullContextEngine protocol slot; Granite is
        // self-contained (tokens.json + granite.json inside the asset).
        _ = piecesURL
        let manifestURL = modelURL.appendingPathComponent("granite.json")
        guard let manifestData = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
              (manifest["format"] as? Int) == 1,
              let windowList = manifest["windows"] as? [[String: Any]],
              let blank = manifest["blank_id"] as? Int,
              let minEdit = manifest["min_edit_sequence_length"] as? Int
        else {
            throw ASRError.modelLoadFailed("missing/malformed granite.json in \(modelURL.path)")
        }
        self.blankId = Int32(blank)
        self.minEditLength = minEdit
        self.windows = windowList.compactMap { w in
            guard let f = w["frames"] as? Int, let a = w["audio_len"] as? Int,
                  let u = w["text_bucket"] as? Int else { return nil }
            return Window(frames: f, audioLen: a, textBucket: u)
        }.sorted { $0.frames < $1.frames }
        guard !windows.isEmpty else {
            throw ASRError.modelLoadFailed("granite.json has no windows")
        }
        self.detokenizer = try GraniteBPEDetokenizer(
            tokensURL: modelURL.appendingPathComponent("tokens.json"))

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

    /// Background-load the smallest window's encode+edit pair so weight
    /// materialization overlaps audio capture. `VOIXFUL_PREWARM=0` disables.
    public func prewarm() {
        guard ProcessInfo.processInfo.environment["VOIXFUL_PREWARM"] != "0" else { return }
        guard let smallest = windows.first else { return }
        let model = self.model
        for name in [smallest.encodeName, smallest.editName] where loaded[name] == nil && pending[name] == nil {
            pending[name] = Task.detached(priority: .utility) {
                (try? model.loadFunction(named: name)) ?? nil
            }
        }
    }

    public func transcribe(audio: [Float]) async throws -> Transcript {
        let t0 = CFAbsoluteTimeGetCurrent()
        let realFrames = (audio.count + Self.samplesPerFrame - 1) / Self.samplesPerFrame
        let window = windows.first(where: { $0.frames >= realFrames }) ?? windows[windows.count - 1]
        let samples = window.frames * Self.samplesPerFrame

        var audioND = NDArray(shape: [1, samples], scalarType: .float32)
        var audioMV = audioND.mutableView(as: Float.self)
        audioMV.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: 0, count: samples)
            audio.withUnsafeBufferPointer { src in
                ptr.update(from: src.baseAddress!, count: min(audio.count, samples))
            }
        }

        let encode = try await fn(window.encodeName)
        let tLoad = CFAbsoluteTimeGetCurrent()
        var encOut = try await encode.run(
            inputs: ["audio": audioND],
            states: InferenceFunction.MutableViews())
        guard let audioEmbeds = encOut.remove("audio_embeds")?.ndArray,
              let ctcND = encOut.remove("ctc_ids")?.ndArray else {
            throw ASRError.missingOutput("encode outputs")
        }
        let tEnc = CFAbsoluteTimeGetCurrent()

        // CTC collapse: unique-consecutive over the id sequence, drop blank.
        var hypothesis: [Int32] = []
        let ctcCount = ctcND.shape.reduce(1, *)
        ctcND.view(as: Int32.self).withUnsafePointer { ptr, _, _ in
            var prev: Int32 = -1
            for i in 0..<ctcCount {
                let id = ptr[i]
                if id != prev, id != blankId { hypothesis.append(id) }
                prev = id
            }
        }
        // Insertion-slot interleave, clamped to the window's bucket.
        let maxTokens = (window.textBucket - 1) / 2
        if hypothesis.count > maxTokens { hypothesis.removeLast(hypothesis.count - maxTokens) }
        let editLen = max(2 * hypothesis.count + 1, minEditLength)

        var textND = NDArray(shape: [1, window.textBucket], scalarType: .int32)
        var textMV = textND.mutableView(as: Int32.self)
        textMV.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: blankId, count: window.textBucket)
            for (i, tok) in hypothesis.enumerated() { ptr[2 * i + 1] = tok }
        }
        let maskLen = window.audioLen + window.textBucket
        var maskND = NDArray(shape: [1, maskLen], scalarType: .float32)
        var maskMV = maskND.mutableView(as: Float.self)
        maskMV.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: 0, count: maskLen)
            for i in 0..<(window.audioLen + editLen) { ptr[i] = 1 }
        }

        let edit = try await fn(window.editName)
        var editOut = try await edit.run(
            inputs: ["audio_embeds": audioEmbeds, "text_ids": textND, "attn_mask": maskND],
            states: InferenceFunction.MutableViews())
        guard let idsND = editOut.remove("ids")?.ndArray else {
            throw ASRError.missingOutput("ids")
        }
        let tEdit = CFAbsoluteTimeGetCurrent()

        var final: [Int32] = []
        idsND.view(as: Int32.self).withUnsafePointer { ptr, _, _ in
            var prev: Int32 = -1
            for i in 0..<editLen {
                let id = ptr[i]
                if id != prev, id != blankId { final.append(id) }
                prev = id
            }
        }
        let text = detokenizer.decode(final)

        if Self.trace {
            let line = String(
                format: "[granite/timing] audio=%.1fs frames=%d (window=%d) | "
                    + "load=%.0fms encode=%.0fms edit=%.0fms | ctc=%d tokens, final=%d tokens\n",
                Double(audio.count) / 16_000, realFrames, window.frames,
                (tLoad - t0) * 1000, (tEnc - tLoad) * 1000, (tEdit - tEnc) * 1000,
                hypothesis.count, final.count)
            FileHandle.standardError.write(Data(line.utf8))
        }
        return Transcript(text: text, segments: [], language: nil)
    }
}
