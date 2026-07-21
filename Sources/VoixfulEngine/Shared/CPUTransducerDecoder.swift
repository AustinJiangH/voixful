// CPU-side transducer decode (prediction LSTM + TDT joint) via Accelerate.
//
// Measured motivation: transducer decode through Core AI costs ~0.4 ms per
// dispatch regardless of payload, placement, or graph size (see the
// perf-encoder-windows branch log) — ~90 dispatches on a 5.5 s clip = ~37 ms.
// The decode-side weights are only ~8 M params, so running predict/join on
// CPU replaces every dispatch with microsecond-scale sgemv calls — the same
// architecture FluidAudio credits for its Parakeet decode speed.
//
// Weights come from the fp32 sidecar `<asset>.transducer.json/.bin` written
// by `parakeet_convert/dump_transducer.py`. The encoder projection
// (`project_enc`) stays in-graph — one big ANE-friendly matmul per clip; this
// class consumes its output.

import Accelerate
import Foundation

/// Not an actor: owned and driven exclusively by `ParakeetTDTEngine`'s actor.
final class CPUTransducerDecoder {
    let predHidden: Int
    let tokenLogits: Int
    let outDim: Int
    let blankId: Int32
    let durations: [Int]

    // Weights, fp32 row-major as dumped by torch (out_features x in_features).
    private let embed: [Float]                       // [8193 * 640]
    private let wIH: [[Float]], wHH: [[Float]]       // per layer [2560 * 640]
    private let bIH: [[Float]], bHH: [[Float]]       // per layer [2560]
    private let predW: [Float], predB: [Float]       // [640*640], [640]
    private let outW: [Float], outB: [Float]         // [outDim*640], [outDim]
    /// Optional raw-encoder projection (joint.enc). Present for Nemotron,
    /// whose asset has no in-graph project_enc — the CPU decoder projects raw
    /// [encIn] encoder steps itself.
    private let encW: [Float]?, encB: [Float]?
    let encIn: Int
    private var eProj: [Float]

    // LSTM state + current projected prediction embedding.
    private var h: [[Float]], c: [[Float]]           // per layer [640]
    private var predG: [Float]                       // [640] = predW @ h_last + predB

    // Scratch
    private var gates = [Float](repeating: 0, count: 2560)
    private var scratch = [Float](repeating: 0, count: 2560)
    private var z: [Float]
    private var logits: [Float]

    init?(sidecarFor modelURL: URL) {
        let jsonURL = URL(fileURLWithPath: modelURL.path + ".transducer.json")
        let binURL = URL(fileURLWithPath: modelURL.path + ".transducer.bin")
        guard let manifestData = try? Data(contentsOf: jsonURL),
              let manifest = try? JSONSerialization.jsonObject(with: manifestData) as? [String: Any],
              (manifest["format"] as? Int) == 1,
              let blob = try? Data(contentsOf: binURL, options: .mappedIfSafe),
              let tensorList = manifest["tensors"] as? [[String: Any]]
        else { return nil }

        var tensors: [String: [Float]] = [:]
        for t in tensorList {
            guard let name = t["name"] as? String,
                  let offset = t["offset"] as? Int,
                  let count = t["count"] as? Int,
                  offset + count * 4 <= blob.count else { return nil }
            tensors[name] = blob.withUnsafeBytes { raw in
                Array(UnsafeBufferPointer(
                    start: raw.baseAddress!.advanced(by: offset).assumingMemoryBound(to: Float.self),
                    count: count))
            }
        }
        guard let embed = tensors["embed.weight"],
              let predW = tensors["joint.pred.weight"], let predB = tensors["joint.pred.bias"],
              let outW = tensors["joint.out.weight"], let outB = tensors["joint.out.bias"]
        else { return nil }
        var wIH: [[Float]] = [], wHH: [[Float]] = [], bIH: [[Float]] = [], bHH: [[Float]] = []
        for l in 0..<((manifest["lstm_layers"] as? Int) ?? 2) {
            guard let a = tensors["lstm.l\(l).w_ih"], let b = tensors["lstm.l\(l).w_hh"],
                  let d = tensors["lstm.l\(l).b_ih"], let e = tensors["lstm.l\(l).b_hh"]
            else { return nil }
            wIH.append(a); wHH.append(b); bIH.append(d); bHH.append(e)
        }

        let ph = (manifest["pred_hidden"] as? Int) ?? 640
        self.predHidden = ph
        self.encW = tensors["joint.enc.weight"]
        self.encB = tensors["joint.enc.bias"]
        self.encIn = (manifest["enc_in"] as? Int) ?? ph
        self.eProj = [Float](repeating: 0, count: ph)
        self.tokenLogits = (manifest["token_logits"] as? Int) ?? 8193
        self.blankId = Int32((manifest["blank_id"] as? Int) ?? 8192)
        self.durations = (manifest["durations"] as? [Int]) ?? [0, 1, 2, 3, 4]
        self.outDim = outB.count
        self.embed = embed
        self.wIH = wIH; self.wHH = wHH; self.bIH = bIH; self.bHH = bHH
        self.predW = predW; self.predB = predB
        self.outW = outW; self.outB = outB
        let zeros = [Float](repeating: 0, count: ph)
        self.h = wIH.map { _ in zeros }
        self.c = wIH.map { _ in zeros }
        self.predG = zeros
        self.z = zeros
        self.logits = [Float](repeating: 0, count: outB.count)
    }

    /// Zero the LSTM state, then run the blank-as-SOS step (blank is the
    /// embedding's padding_idx, so its row is zeros — LSTM(zeros, zero state)
    /// is exactly NeMo's g_0 seeding).
    func reset() {
        for l in 0..<h.count {
            h[l].withUnsafeMutableBufferPointer { vDSP.fill(&$0, with: 0) }
            c[l].withUnsafeMutableBufferPointer { vDSP.fill(&$0, with: 0) }
        }
        step(token: blankId)
    }

    /// Advance the prediction LSTM by one token and refresh `predG`.
    func step(token: Int32) {
        let H = predHidden
        var x = [Float](repeating: 0, count: H)
        let row = Int(token) * H
        if token != blankId {  // blank row is zeros (padding_idx)
            embed.withUnsafeBufferPointer { e in
                x.withUnsafeMutableBufferPointer { xb in
                    xb.baseAddress!.update(from: e.baseAddress! + row, count: H)
                }
            }
        }
        for l in 0..<h.count {
            // gates = b_ih + b_hh + W_ih@x + W_hh@h
            vDSP_vadd(bIH[l], 1, bHH[l], 1, &gates, 1, vDSP_Length(4 * H))
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(4 * H), Int32(H),
                        1, wIH[l], Int32(H), x, 1, 1, &gates, 1)
            cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(4 * H), Int32(H),
                        1, wHH[l], Int32(H), h[l], 1, 1, &gates, 1)
            // PyTorch gate order: i, f, g, o.
            var n = Int32(2 * H)
            // sigmoid on [i|f] in place: 1/(1+exp(-x))
            vDSP_vneg(gates, 1, &scratch, 1, vDSP_Length(2 * H))
            vvexpf(&scratch, scratch, &n)
            var one: Float = 1
            vDSP_vsadd(scratch, 1, &one, &scratch, 1, vDSP_Length(2 * H))
            var n2 = Int32(2 * H)
            vvrecf(&gates, scratch, &n2)      // gates[0..2H) = sigmoid(i|f)
            var nH = Int32(H)
            var gCell = [Float](repeating: 0, count: H)
            vvtanhf(&gCell, Array(gates[(2 * H)..<(3 * H)]), &nH)
            var o = [Float](repeating: 0, count: H)
            vDSP_vneg(Array(gates[(3 * H)..<(4 * H)]), 1, &scratch, 1, vDSP_Length(H))
            var nO = Int32(H)
            vvexpf(&scratch, scratch, &nO)
            vDSP_vsadd(scratch, 1, &one, &scratch, 1, vDSP_Length(H))
            vvrecf(&o, scratch, &nO)
            // c = f*c + i*g ; h = o * tanh(c)
            vDSP_vmul(Array(gates[H..<(2 * H)]), 1, c[l], 1, &c[l], 1, vDSP_Length(H))
            vDSP_vma(gates, 1, gCell, 1, c[l], 1, &scratch, 1, vDSP_Length(H))
            c[l] = Array(scratch[0..<H])
            var tanhC = [Float](repeating: 0, count: H)
            vvtanhf(&tanhC, c[l], &nH)
            vDSP_vmul(o, 1, tanhC, 1, &h[l], 1, vDSP_Length(H))
            x = h[l]
        }
        // predG = predW @ h_last + predB
        predG = predB
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(H), Int32(H),
                    1, predW, Int32(H), h[h.count - 1], 1, 1, &predG, 1)
    }

    /// Joint for one encoder frame (already `project_enc`-projected):
    /// relu(encProjT + predG) @ outW^T + outB -> (token, durationSkip).
    func join(encProjT: UnsafePointer<Float>) -> (token: Int32, skip: Int) {
        let H = predHidden
        vDSP_vadd(encProjT, 1, predG, 1, &z, 1, vDSP_Length(H))
        var zero: Float = 0
        var inf = Float.greatestFiniteMagnitude
        vDSP_vclip(z, 1, &zero, &inf, &z, 1, vDSP_Length(H))  // ReLU
        logits = outB
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(outDim), Int32(H),
                    1, outW, Int32(H), z, 1, 1, &logits, 1)
        var maxV: Float = 0
        var maxI: vDSP_Length = 0
        vDSP_maxvi(logits, 1, &maxV, &maxI, vDSP_Length(tokenLogits))
        var skip = 0
        if !durations.isEmpty {  // TDT duration head; plain RNN-T has none
            var durIdx = 0
            var durV = -Float.greatestFiniteMagnitude
            for j in 0..<durations.count where logits[tokenLogits + j] > durV {
                durV = logits[tokenLogits + j]
                durIdx = j
            }
            skip = durations[durIdx]
        }
        return (Int32(maxI), skip)
    }

    /// Joint for one RAW encoder frame ([encIn] floats): projects through
    /// joint.enc on CPU first (requires the sidecar to carry it — Nemotron).
    func join(encRaw: UnsafePointer<Float>) -> (token: Int32, skip: Int) {
        guard let encW, let encB else { return join(encProjT: encRaw) }
        eProj = encB
        cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(predHidden), Int32(encIn),
                    1, encW, Int32(encIn), encRaw, 1, 1, &eProj, 1)
        return eProj.withUnsafeBufferPointer { join(encProjT: $0.baseAddress!) }
    }
}
