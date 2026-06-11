// Phase C — Mel preprocessor (Accelerate / vDSP).
// See PLAN.md §5 for the full algorithm and §5 "parity test" for the gate this targets.
// This is the Swift+Accelerate fallback (PLAN.md §5 Branch C). Phase 4 decides whether
// Branches A (in-graph) or B (TorchMetalKernel) supersede it; until then this is the
// source of truth for "what mel features the encoder sees."

import Foundation
import Accelerate

public actor MelExtractor {

    // MARK: - Configuration

    public let sampleRate: Int
    public let nMels: Int
    public let windowSize: Int
    public let hopSize: Int
    public let fftSize: Int
    public let fmin: Float
    public let fmax: Float

    // MARK: - Precomputed state (built once at init)

    /// Real FFT setup; radix-2, length == fftSize. Destroyed in `close()`.
    private let fftSetup: FFTSetup
    /// log2(fftSize), stashed because vDSP_fft_zrip needs it on every call.
    private let log2n: vDSP_Length
    /// Hann window of length `windowSize` (vDSP_HANN_NORM convention).
    private let hannWindow: [Float]
    /// Slaney-style triangular mel filterbank, row-major `[nMels × (fftSize/2 + 1)]`.
    /// NOTE: Slaney mel scale, not HTK — see PLAN.md §5. Librosa's `librosa.filters.mel`
    /// with `htk=False, norm='slaney'` is the reference.
    private let melFilterbank: [Float]
    /// fftSize/2 + 1, the count of non-redundant real-FFT bins.
    private let nFFTBins: Int

    // MARK: - Streaming state

    /// Rolling buffer of samples carried over from prior `extractStreaming` calls.
    private var carryoverSamples: [Float] = []
    /// Welford running mean per mel bin, length `nMels`.
    private var runningMean: [Float]
    /// Welford running variance accumulator (M2) per mel bin, length `nMels`.
    private var runningM2: [Float]
    /// Number of frames seen by the running estimator.
    private var runningCount: Int = 0

    // MARK: - Init

    public init(
        sampleRate: Int = 16_000,
        nMels: Int = 128,
        windowSize: Int = 400,
        hopSize: Int = 160,
        fftSize: Int = 512,
        fmin: Float = 0,
        fmax: Float = 8_000
    ) {
        precondition(fftSize >= windowSize, "fftSize must be >= windowSize")
        precondition(fftSize.nonzeroBitCount == 1, "fftSize must be a power of two for vDSP radix-2")

        self.sampleRate = sampleRate
        self.nMels = nMels
        self.windowSize = windowSize
        self.hopSize = hopSize
        self.fftSize = fftSize
        self.fmin = fmin
        self.fmax = fmax
        self.nFFTBins = fftSize / 2 + 1

        // Hann window — symmetric, NOT normalized. PyTorch's torch.stft / NeMo's
        // FilterbankFeatures use an UN-normalized Hann (peak amplitude 1.0). Apple's
        // vDSP_HANN_NORM divides by sum-of-window, which gave us ~14 nats too much
        // power in log space. Use 0 (no flag) for the un-normalized form that matches.
        var window = [Float](repeating: 0, count: windowSize)
        vDSP_hann_window(&window, vDSP_Length(windowSize), 0)
        self.hannWindow = window

        // Slaney triangular filterbank.
        self.melFilterbank = Self.buildSlaneyFilterbank(
            nMels: nMels,
            nFFTBins: nFFTBins,
            sampleRate: sampleRate,
            fmin: fmin,
            fmax: fmax
        )

        let log2n = vDSP_Length(log2(Float(fftSize)).rounded())
        self.log2n = log2n
        guard let setup = vDSP_create_fftsetup(log2n, Int32(kFFTRadix2)) else {
            preconditionFailure("vDSP_create_fftsetup returned nil for log2n=\(log2n)")
        }
        self.fftSetup = setup

        self.runningMean = [Float](repeating: 0, count: nMels)
        self.runningM2 = [Float](repeating: 0, count: nMels)
    }

    // Resolves the Phase-C VERIFY: Swift 6 / SDK 27.0 confirms actor deinit
    // cannot access non-Sendable FFTSetup (OpaquePointer). Migrated to explicit
    // `close()` as the Phase-C reviewer anticipated. Callers (StreamingASR)
    // invoke `await mel.close()` on shutdown; if a session is dropped without
    // close, vDSP's setup leaks but the leak is bounded (one allocation per
    // MelExtractor lifetime, typically one per app).
    //
    // `closed` guards against double-destroy if a future caller invokes close()
    // explicitly and then a stray reference triggers another close().
    private var closed = false

    /// Release the underlying vDSP FFT setup. Idempotent. Call once at the end
    /// of the MelExtractor's useful life (e.g., when StreamingASR is being
    /// torn down).
    public func close() {
        guard !closed else { return }
        vDSP_destroy_fftsetup(fftSetup)
        closed = true
    }

    // MARK: - Public API

    public func reset() {
        carryoverSamples.removeAll(keepingCapacity: true)
        for i in 0..<nMels {
            runningMean[i] = 0
            runningM2[i] = 0
        }
        runningCount = 0
    }

    /// Non-streaming: process the entire buffer, return row-major `[nMels × T]` flattened.
    public func extract(samples: [Float]) -> [Float] {
        guard samples.count >= windowSize else { return [] }
        let nFrames = (samples.count - windowSize) / hopSize + 1
        var out = [Float](repeating: 0, count: nMels * nFrames)
        var frame = [Float](repeating: 0, count: nMels)

        for t in 0..<nFrames {
            let start = t * hopSize
            let end = start + windowSize
            runFrame(samples: samples, start: start, end: end, into: &frame)
            // NeMo's preprocessor for this checkpoint has `normalize: 'NA'` —
            // no per-feature normalization. Applying Welford here gave garbage
            // output. Keep the running-stats path warm in case a future
            // checkpoint enables normalization, but skip applying it.
            // Row-major store: row m at offset m * nFrames + t. T is the column index.
            for m in 0..<nMels {
                out[m * nFrames + t] = frame[m]
            }
        }
        return out
    }

    /// Streaming: append `samples` to the carryover, process every full hop available,
    /// return the consumed frames row-major `[nMels × T_consumed]` flattened. Returns
    /// `nil` when no full frame can be produced yet.
    public func extractStreaming(samples: [Float]) -> [Float]? {
        carryoverSamples.append(contentsOf: samples)
        guard carryoverSamples.count >= windowSize else { return nil }

        let totalFrames = (carryoverSamples.count - windowSize) / hopSize + 1
        guard totalFrames > 0 else { return nil }

        var out = [Float](repeating: 0, count: nMels * totalFrames)
        var frame = [Float](repeating: 0, count: nMels)

        for t in 0..<totalFrames {
            let start = t * hopSize
            let end = start + windowSize
            runFrame(samples: carryoverSamples, start: start, end: end, into: &frame)
            // Match NeMo's `normalize: 'NA'` config — no normalization. See
            // the non-streaming path's comment.
            for m in 0..<nMels {
                out[m * totalFrames + t] = frame[m]
            }
        }

        // Drop everything we've consumed; keep the tail needed to overlap the next frame.
        let consumed = totalFrames * hopSize
        let keepFrom = consumed
        let tailCount = carryoverSamples.count - keepFrom
        if tailCount > 0 {
            carryoverSamples = Array(carryoverSamples[keepFrom..<carryoverSamples.count])
        } else {
            carryoverSamples.removeAll(keepingCapacity: true)
        }
        return out
    }

    // MARK: - Per-frame pipeline

    /// STFT → power → mel → log for a single frame. `frame` must be `nMels` long; on return
    /// it holds the (un-normalized) log-mel vector for one time step.
    private func runFrame(samples: [Float], start: Int, end: Int, into frame: inout [Float]) {
        precondition(end - start == windowSize)
        precondition(frame.count == nMels)

        // 1) Window: windowed[i] = samples[start+i] * hannWindow[i], zero-padded to fftSize.
        var windowed = [Float](repeating: 0, count: fftSize)
        samples.withUnsafeBufferPointer { src in
            hannWindow.withUnsafeBufferPointer { win in
                let base = src.baseAddress!.advanced(by: start)
                vDSP_vmul(base, 1, win.baseAddress!, 1, &windowed, 1, vDSP_Length(windowSize))
            }
        }

        // 2) Real FFT via the vDSP_fft_zrip "split-complex packed" convention.
        // For a real input of length N, vDSP packs into N/2 complex bins where
        // real[0] holds DC, imag[0] holds Nyquist, and bins 1..N/2-1 are the usual
        // half-spectrum. We un-pack into a power-spectrum of length N/2+1 below.
        let half = fftSize / 2
        var realp = [Float](repeating: 0, count: half)
        var imagp = [Float](repeating: 0, count: half)
        var powerSpectrum = [Float](repeating: 0, count: nFFTBins)

        realp.withUnsafeMutableBufferPointer { rp in
            imagp.withUnsafeMutableBufferPointer { ip in
                var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)

                // Pack interleaved real input into split-complex form.
                windowed.withUnsafeBufferPointer { w in
                    w.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) { complexPtr in
                        vDSP_ctoz(complexPtr, 2, &split, 1, vDSP_Length(half))
                    }
                }

                vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(FFT_FORWARD))

                // Power = magnitude squared. zrip's scaling is 2x the mathematical DFT;
                // for a power spectrum used as filterbank input, the constant scale is
                // absorbed by the log + normalization so we don't compensate here.
                // VERIFY: NeMo's preprocessor applies a 1/N normalization on the magnitude
                // squared (librosa default). If parity drift shows a constant log offset,
                // multiply powerSpectrum by 1.0 / Float(fftSize * fftSize). See PLAN.md §5.
                var midPower = [Float](repeating: 0, count: half)
                vDSP_zvmags(&split, 1, &midPower, 1, vDSP_Length(half))

                // DC bin is split.realp[0]; Nyquist is split.imagp[0] (packed slot).
                let dc = rp.baseAddress![0]
                let nyq = ip.baseAddress![0]
                powerSpectrum[0] = dc * dc
                powerSpectrum[nFFTBins - 1] = nyq * nyq
                // Bins 1..half-1 come straight from vDSP_zvmags (skip slot 0 which was DC+Nyquist).
                for k in 1..<half {
                    powerSpectrum[k] = midPower[k]
                }
                // NeMo's FilterbankFeatures normalizes the power spectrum by
                // (sum(window))^2 in NumPy/torchaudio convention. For a Hann window
                // of length 400 the sum is ~200, square ~4e4. Combined with Apple's
                // 2x FFT scaling (4x in power), the effective divisor is 4 * 200^2 ≈
                // 1.6e5 = log offset ~11.95 — matches the residual drift after the
                // 0.25 fix. Apply both as one multiplier.
                let windowSum = hannWindow.reduce(0, +)
                var invScale: Float = 0.25 / (windowSum * windowSum)
                vDSP_vsmul(powerSpectrum, 1, &invScale, &powerSpectrum, 1, vDSP_Length(nFFTBins))
            }
        }

        // 3) Mel projection: melFilterbank [nMels × nFFTBins] · powerSpectrum [nFFTBins × 1].
        melFilterbank.withUnsafeBufferPointer { fb in
            powerSpectrum.withUnsafeBufferPointer { ps in
                vDSP_mmul(
                    fb.baseAddress!, 1,
                    ps.baseAddress!, 1,
                    &frame, 1,
                    vDSP_Length(nMels), 1, vDSP_Length(nFFTBins)
                )
            }
        }

        // 4) log(x + 2^-24), elementwise. NeMo's FilterbankFeatures uses
        // log_zero_guard_value = 2**(-24) ≈ 5.96e-8, not the 1e-5 PLAN.md
        // wrote down. The smaller offset matters: it controls the log floor
        // for near-silent frames. With 1e-5 our mel values landed in roughly
        // [-12, 0]; the reference is closer to [-16.6, 0.7]. Aligning the
        // offset brings them into the same range.
        var offset: Float = Float(1.0 / Float(1 << 24))
        vDSP_vsadd(frame, 1, &offset, &frame, 1, vDSP_Length(nMels))
        var count = Int32(nMels)
        vvlogf(&frame, frame, &count)
    }

    // MARK: - Running per-feature normalization (Welford)
    // VERIFY: confirm normalization mode (per_feature vs all_features, fixed vs running
    // estimator) against PLAN.md §10 open question — Phase 1 task. NeMo's
    // `AudioToMelSpectrogramPreprocessor` supports both `per_feature` and `all_features`;
    // streaming likely wants a windowed running estimator over the last ~3 s, not a
    // global accumulator. The Welford form below is a global running estimator and is a
    // placeholder until the checkpoint's preprocessor config is inspected.

    private func updateRunningStats(frame: [Float]) {
        precondition(frame.count == nMels)
        runningCount += 1
        let count = Float(runningCount)
        for i in 0..<nMels {
            let x = frame[i]
            let delta = x - runningMean[i]
            runningMean[i] += delta / count
            let delta2 = x - runningMean[i]
            runningM2[i] += delta * delta2
        }
    }

    private func normalize(frame: inout [Float]) {
        guard runningCount > 1 else {
            // First frame: subtract mean only; variance is undefined with one sample.
            for i in 0..<nMels { frame[i] -= runningMean[i] }
            return
        }
        let denom = Float(runningCount - 1)
        for i in 0..<nMels {
            let variance = runningM2[i] / denom
            let std = sqrt(variance + 1e-5)
            frame[i] = (frame[i] - runningMean[i]) / std
        }
    }

    // MARK: - Slaney filterbank construction

    /// Build a Slaney-style triangular mel filterbank, row-major `[nMels × nFFTBins]`.
    /// Reference: librosa.filters.mel(..., htk=False, norm='slaney').
    private static func buildSlaneyFilterbank(
        nMels: Int,
        nFFTBins: Int,
        sampleRate: Int,
        fmin: Float,
        fmax: Float
    ) -> [Float] {
        let melMin = hzToMelSlaney(fmin)
        let melMax = hzToMelSlaney(fmax)

        // nMels + 2 mel-spaced edge points → nMels triangular filters.
        var melPoints = [Float](repeating: 0, count: nMels + 2)
        let step = (melMax - melMin) / Float(nMels + 1)
        for i in 0..<(nMels + 2) {
            melPoints[i] = melMin + step * Float(i)
        }
        let hzPoints = melPoints.map { melToHzSlaney($0) }

        // FFT bin centers in Hz: bin k corresponds to k * sampleRate / fftSize.
        // nFFTBins == fftSize/2 + 1, so fftSize = 2 * (nFFTBins - 1).
        let fftSize = 2 * (nFFTBins - 1)
        var binFreqs = [Float](repeating: 0, count: nFFTBins)
        for k in 0..<nFFTBins {
            binFreqs[k] = Float(k) * Float(sampleRate) / Float(fftSize)
        }

        var filterbank = [Float](repeating: 0, count: nMels * nFFTBins)
        for m in 0..<nMels {
            let lo = hzPoints[m]
            let center = hzPoints[m + 1]
            let hi = hzPoints[m + 2]
            // Slaney normalization: scale each filter by 2 / (hi - lo) so triangular
            // filters integrate to a constant over their support.
            let enorm = 2.0 / (hi - lo)
            for k in 0..<nFFTBins {
                let f = binFreqs[k]
                let upSlope = (f - lo) / (center - lo)
                let downSlope = (hi - f) / (hi - center)
                let w = max(0, min(upSlope, downSlope))
                filterbank[m * nFFTBins + k] = w * enorm
            }
        }
        return filterbank
    }

    /// Slaney mel scale: linear below 1 kHz, log above. Matches librosa `htk=False`.
    private static func hzToMelSlaney(_ hz: Float) -> Float {
        let fMin: Float = 0
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1_000
        let minLogMel = (minLogHz - fMin) / fSp
        let logStep = log(Float(6.4)) / 27.0
        if hz < minLogHz {
            return (hz - fMin) / fSp
        }
        return minLogMel + log(hz / minLogHz) / logStep
    }

    private static func melToHzSlaney(_ mel: Float) -> Float {
        let fMin: Float = 0
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1_000
        let minLogMel = (minLogHz - fMin) / fSp
        let logStep = log(Float(6.4)) / 27.0
        if mel < minLogMel {
            return fMin + fSp * mel
        }
        return minLogHz * exp(logStep * (mel - minLogMel))
    }
}
