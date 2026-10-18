// Core AI runtime for IBM Granite Speech 5.0 TurboCTC (470M, English).
//
// The simplest backend Voixful runs: ONE graph per audio window and no decode
// loop. `ctc_<F>` takes raw 16 kHz audio padded to the window plus its real
// sample count, and returns per-frame argmax ids at 12.5 Hz (blank past the
// real length — the count drives the encoder's padding mask in-graph, so a
// padded window decodes exactly like the unpadded clip). The host only does
// the CTC collapse (unique-consecutive, drop blank) and the byte-level BPE
// detokenize (shared with Granite NAR).
//
// Output is lowercase without punctuation (plain CTC; numbers are normalized,
// e.g. "12%"). Window table and blank id come from the asset's turboctc.json;
// tokens.json is the per-id raw-byte table (both bundled at convert time, so
// the specialize cache keys on the final asset contents).

import CoreAI
import Foundation

public actor TurboCTCEngine {
    private let model: AIModel
    private let detokenizer: GraniteBPEDetokenizer
    private var loaded: [String: InferenceFunction] = [:]
    private var pending: [String: Task<InferenceFunction?, Never>] = [:]

    struct Window {
        let frames: Int  // stacked 50 Hz input frames; samples = frames * 320
        var name: String { "ctc_\(frames)" }
    }

    private let windows: [Window]  // ascending by frames
    private let blankId: Int32
    private let samplesPerFrame: Int

    /// Largest single-pass audio: the biggest window's sample count.
    public nonisolated var maxAudioSamples: Int {
        (windows.last?.frames ?? 1536) * samplesPerFrame
    }

    private static let trace = ProcessInfo.processInfo.environment["TURBOCTC_TRACE"] != nil

    public init(modelURL: URL, piecesURL: URL) async throws {
        // piecesURL is the FullContextEngine protocol slot; TurboCTC is
        // self-contained (tokens.json + turboctc.json inside the asset).
        _ = piecesURL
        let manifestURL = modelURL.appendingPathComponent("turboctc.json")
        guard let manifestData = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
              (manifest["format"] as? Int) == 1,
              let windowList = manifest["windows"] as? [[String: Any]],
              let blank = manifest["blank_id"] as? Int
        else {
            throw ASRError.modelLoadFailed("missing/malformed turboctc.json in \(modelURL.path)")
        }
        self.blankId = Int32(blank)
        self.samplesPerFrame = manifest["samples_per_frame"] as? Int ?? 320
        self.windows = windowList.compactMap { w in
            (w["frames"] as? Int).map(Window.init(frames:))
        }.sorted { $0.frames < $1.frames }
        guard !windows.isEmpty else {
            throw ASRError.modelLoadFailed("turboctc.json has no windows")
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

    /// Background-load the smallest window so weight materialization overlaps
    /// audio capture. `VOIXFUL_PREWARM=0` disables.
    public func prewarm() {
        guard ProcessInfo.processInfo.environment["VOIXFUL_PREWARM"] != "0" else { return }
        guard let name = windows.first?.name, loaded[name] == nil, pending[name] == nil else { return }
        let model = self.model
        pending[name] = Task.detached(priority: .utility) {
            (try? model.loadFunction(named: name)) ?? nil
        }
    }

    public func transcribe(audio: [Float]) async throws -> Transcript {
        let t0 = CFAbsoluteTimeGetCurrent()
        let realFrames = (audio.count + samplesPerFrame - 1) / samplesPerFrame
        let window = windows.first(where: { $0.frames >= realFrames }) ?? windows[windows.count - 1]
        let samples = window.frames * samplesPerFrame
        let used = min(audio.count, samples)

        var audioND = NDArray(shape: [1, samples], scalarType: .float32)
        var audioMV = audioND.mutableView(as: Float.self)
        audioMV.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: 0, count: samples)
            audio.withUnsafeBufferPointer { src in
                ptr.update(from: src.baseAddress!, count: used)
            }
        }
        var lengthND = NDArray(shape: [1], scalarType: .int32)
        var lengthMV = lengthND.mutableView(as: Int32.self)
        lengthMV.withUnsafeMutablePointer { ptr, _, _ in ptr[0] = Int32(used) }

        let ctc = try await fn(window.name)
        let tLoad = CFAbsoluteTimeGetCurrent()
        var out = try await ctc.run(
            inputs: ["audio": audioND, "length": lengthND],
            states: InferenceFunction.MutableViews())
        guard let idsND = out.remove("ids")?.ndArray else {
            throw ASRError.missingOutput("ids")
        }
        let tRun = CFAbsoluteTimeGetCurrent()

        // CTC collapse: unique-consecutive over the id sequence, drop blank.
        var tokens: [Int32] = []
        let count = idsND.shape.reduce(1, *)
        idsND.view(as: Int32.self).withUnsafePointer { ptr, _, _ in
            var prev: Int32 = -1
            for i in 0..<count {
                let id = ptr[i]
                if id != prev, id != blankId { tokens.append(id) }
                prev = id
            }
        }
        let text = detokenizer.decode(tokens).trimmingCharacters(in: .whitespacesAndNewlines)

        if Self.trace {
            let line = String(
                format: "[turboctc/timing] audio=%.1fs (window=%d) | load=%.0fms run=%.0fms | %d tokens\n",
                Double(audio.count) / 16_000, window.frames,
                (tLoad - t0) * 1000, (tRun - tLoad) * 1000, tokens.count)
            FileHandle.standardError.write(Data(line.utf8))
        }
        return Transcript(text: text, segments: [], language: nil)
    }
}
