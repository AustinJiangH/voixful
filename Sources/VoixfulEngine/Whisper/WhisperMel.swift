// Whisper log-mel front-end (Accelerate / vDSP).
//
// Whisper's mel is NOT the NeMo per-feature mel the shared `MelExtractor`
// produces, so this is a dedicated, self-contained front-end. The differences
// that matter (verified against openai/whisper `audio.log_mel_spectrogram`):
//
//   - n_fft = 400 (NOT a power of two) — the shared vDSP radix-2 FFT can't do
//     it, so we run the STFT as an explicit DFT matmul (cos/sin basis × framed
//     window). 400 = 16·25, ~482 M mults for a 30 s clip — sub-second, done in
//     two `vDSP_mmul`s.
//   - NO pre-emphasis (NeMo/Cohere/Parakeet apply 0.97; Whisper does not).
//   - hop 160, periodic Hann of length 400, center=True (reflect pad n_fft/2).
//   - Slaney mel filterbank, fmin=0 fmax=8000, 80 or 128 channels
//     (librosa.filters.mel(sr=16000, n_fft=400, htk=False, norm='slaney')).
//   - Global (NOT per-feature) normalization: log10, clamp to 1e-10, dynamic
//     range clamp to (max - 8), then (x + 4) / 4.
//
// The input is always padded/truncated to a fixed 30 s (`chunkSamples`) before
// mel, so the output is a fixed `[nMels × melFrames]` (melFrames = 3000).

import Accelerate
import Foundation

/// Computes Whisper's fixed-window log-mel spectrogram. Immutable / value type —
/// the precomputed basis + filterbank are built once at init.
struct WhisperMel {
    let sampleRate: Int
    let nMels: Int
    let nFFT: Int
    let hop: Int
    let chunkSamples: Int   // 30 s at 16 kHz = 480000
    let melFrames: Int      // chunkSamples / hop = 3000

    private let nBins: Int          // nFFT/2 + 1
    private let hann: [Float]       // length nFFT
    private let cosBasis: [Float]   // [nBins × nFFT] row-major
    private let sinBasis: [Float]   // [nBins × nFFT] row-major
    private let melFB: [Float]      // [nMels × nBins] row-major

    init(sampleRate: Int, nMels: Int, nFFT: Int, hop: Int, chunkSamples: Int) {
        self.sampleRate = sampleRate
        self.nMels = nMels
        self.nFFT = nFFT
        self.hop = hop
        self.chunkSamples = chunkSamples
        self.melFrames = chunkSamples / hop
        self.nBins = nFFT / 2 + 1

        // Periodic Hann (torch.hann_window default): w[n] = 0.5·(1 - cos(2πn/N)).
        var w = [Float](repeating: 0, count: nFFT)
        for n in 0..<nFFT {
            w[n] = 0.5 * (1 - cos(2 * Float.pi * Float(n) / Float(nFFT)))
        }
        self.hann = w

        // DFT basis for the real half-spectrum: X[k] = Σ x[n]·exp(-i2πkn/N).
        // We store cos/sin separately; power = Re² + Im² so the Im sign is moot.
        var cb = [Float](repeating: 0, count: nBins * nFFT)
        var sb = [Float](repeating: 0, count: nBins * nFFT)
        for k in 0..<nBins {
            for n in 0..<nFFT {
                let angle = 2 * Float.pi * Float(k) * Float(n) / Float(nFFT)
                cb[k * nFFT + n] = cos(angle)
                sb[k * nFFT + n] = sin(angle)
            }
        }
        self.cosBasis = cb
        self.sinBasis = sb

        self.melFB = Self.slaneyFilterbank(
            nMels: nMels, nBins: nBins, sampleRate: sampleRate,
            fmin: 0, fmax: Float(sampleRate) / 2)
    }

    /// Log-mel spectrogram for `samples`, row-major `[nMels × melFrames]`
    /// flattened. Input is padded/truncated to the fixed 30 s window first.
    func logMel(samples: [Float]) -> [Float] {
        // 1. Pad/trim to the fixed 30 s window (Whisper pad_or_trim).
        var audio = [Float](repeating: 0, count: chunkSamples)
        let copy = min(samples.count, chunkSamples)
        if copy > 0 {
            samples.withUnsafeBufferPointer { src in
                audio.withUnsafeMutableBufferPointer { dst in
                    dst.baseAddress!.update(from: src.baseAddress!, count: copy)
                }
            }
        }

        // 2. Reflect pad by nFFT/2 each side (torch.stft center=True). Reflection
        //    skips the boundary sample: [a,b,c] pad 2 -> [c,b,a,b,c,...].
        let pad = nFFT / 2
        var padded = [Float](repeating: 0, count: chunkSamples + 2 * pad)
        for i in 0..<chunkSamples { padded[pad + i] = audio[i] }
        for k in 1...pad {
            padded[pad - k] = audio[k]                       // left reflect
            padded[pad + chunkSamples - 1 + k] = audio[chunkSamples - 1 - k]  // right
        }

        // 3. Framed windowed matrix F [nFFT × melFrames]: column t = padded frame
        //    starting at t·hop, times the Hann window. (magnitudes = stft[:, :-1],
        //    so we emit exactly `melFrames` frames.)
        var framed = [Float](repeating: 0, count: nFFT * melFrames)
        padded.withUnsafeBufferPointer { p in
            hann.withUnsafeBufferPointer { hw in
                for t in 0..<melFrames {
                    let base = t * hop
                    for n in 0..<nFFT {
                        framed[n * melFrames + t] = p[base + n] * hw[n]
                    }
                }
            }
        }

        // 4. DFT via matmul: Re/Im = basis[nBins×nFFT] · F[nFFT×melFrames].
        var re = [Float](repeating: 0, count: nBins * melFrames)
        var im = [Float](repeating: 0, count: nBins * melFrames)
        cosBasis.withUnsafeBufferPointer { cb in
            framed.withUnsafeBufferPointer { f in
                vDSP_mmul(cb.baseAddress!, 1, f.baseAddress!, 1, &re, 1,
                          vDSP_Length(nBins), vDSP_Length(melFrames), vDSP_Length(nFFT))
            }
        }
        sinBasis.withUnsafeBufferPointer { sb in
            framed.withUnsafeBufferPointer { f in
                vDSP_mmul(sb.baseAddress!, 1, f.baseAddress!, 1, &im, 1,
                          vDSP_Length(nBins), vDSP_Length(melFrames), vDSP_Length(nFFT))
            }
        }

        // 5. Power spectrum |X|² = Re² + Im².
        var power = [Float](repeating: 0, count: nBins * melFrames)
        vDSP_vsq(re, 1, &re, 1, vDSP_Length(re.count))
        vDSP_vsq(im, 1, &im, 1, vDSP_Length(im.count))
        vDSP_vadd(re, 1, im, 1, &power, 1, vDSP_Length(power.count))

        // 6. Mel projection: mel[nMels×melFrames] = melFB[nMels×nBins] · power.
        var mel = [Float](repeating: 0, count: nMels * melFrames)
        melFB.withUnsafeBufferPointer { fb in
            power.withUnsafeBufferPointer { ps in
                vDSP_mmul(fb.baseAddress!, 1, ps.baseAddress!, 1, &mel, 1,
                          vDSP_Length(nMels), vDSP_Length(melFrames), vDSP_Length(nBins))
            }
        }

        // 7. Whisper's log + normalization: log10(clamp(mel, 1e-10)); dynamic
        //    range clamp to (globalMax - 8); (x + 4) / 4.
        var floorV: Float = 1e-10
        vDSP_vthr(mel, 1, &floorV, &mel, 1, vDSP_Length(mel.count))  // max(mel, 1e-10)
        var count = Int32(mel.count)
        vvlog10f(&mel, mel, &count)
        var globalMax: Float = -.infinity
        vDSP_maxv(mel, 1, &globalMax, vDSP_Length(mel.count))
        var lowerBound = globalMax - 8
        vDSP_vthr(mel, 1, &lowerBound, &mel, 1, vDSP_Length(mel.count))  // max(x, max-8)
        var add: Float = 4
        var scale: Float = 0.25
        vDSP_vsadd(mel, 1, &add, &mel, 1, vDSP_Length(mel.count))
        vDSP_vsmul(mel, 1, &scale, &mel, 1, vDSP_Length(mel.count))
        return mel
    }

    // MARK: - Slaney mel filterbank (librosa htk=False, norm='slaney')

    /// Row-major `[nMels × nBins]`. Same construction the shared `MelExtractor`
    /// uses, duplicated here so the Whisper front-end stays self-contained.
    private static func slaneyFilterbank(
        nMels: Int, nBins: Int, sampleRate: Int, fmin: Float, fmax: Float
    ) -> [Float] {
        let melMin = hzToMel(fmin)
        let melMax = hzToMel(fmax)
        var melPoints = [Float](repeating: 0, count: nMels + 2)
        let step = (melMax - melMin) / Float(nMels + 1)
        for i in 0..<(nMels + 2) { melPoints[i] = melMin + step * Float(i) }
        let hz = melPoints.map { melToHz($0) }

        let fftSize = 2 * (nBins - 1)
        var binFreqs = [Float](repeating: 0, count: nBins)
        for k in 0..<nBins { binFreqs[k] = Float(k) * Float(sampleRate) / Float(fftSize) }

        var fb = [Float](repeating: 0, count: nMels * nBins)
        for m in 0..<nMels {
            let lo = hz[m], center = hz[m + 1], hi = hz[m + 2]
            let enorm = 2.0 / (hi - lo)
            for k in 0..<nBins {
                let f = binFreqs[k]
                let up = (f - lo) / (center - lo)
                let down = (hi - f) / (hi - center)
                fb[m * nBins + k] = max(0, min(up, down)) * enorm
            }
        }
        return fb
    }

    private static func hzToMel(_ hz: Float) -> Float {
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000
        let minLogMel = minLogHz / fSp
        let logStep = log(Float(6.4)) / 27.0
        return hz < minLogHz ? hz / fSp : minLogMel + log(hz / minLogHz) / logStep
    }

    private static func melToHz(_ mel: Float) -> Float {
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000
        let minLogMel = minLogHz / fSp
        let logStep = log(Float(6.4)) / 27.0
        return mel < minLogMel ? fSp * mel : minLogHz * exp(logStep * (mel - minLogMel))
    }
}
