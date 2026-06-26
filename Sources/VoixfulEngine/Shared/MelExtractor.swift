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

    /// Per-frame start offset (samples) added inside `extract`. `torch.stft`
    /// with `win_length < n_fft` and `center=True` places the `windowSize` Hann
    /// CENTERED in the `fftSize` frame — i.e. the window centre sits at the
    /// frame's nominal time. Left-aligning the window (offset 0) shifts every
    /// frame earlier by `(fftSize - windowSize)/2` samples. The streaming
    /// Nemotron path uses 0 (its existing, verified behaviour); the full-context
    /// Parakeet path passes `(fftSize - windowSize)/2` to match torch exactly.
    public let frameStartOffset: Int

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

    /// Rolling buffer of preemphasized samples carried over from prior `extractStreaming` calls.
    private var carryoverSamples: [Float] = []
    /// Last RAW sample fed in (preemph state); used to apply preemph across
    /// streaming-call boundaries: `y[i] = x[i] - 0.97 * x[i-1]`.
    private var preemphLastRawSample: Float = 0
    /// True once we've seen ANY input; flips the first-sample preemph behavior
    /// (the very first sample of an utterance passes through unchanged).
    private var preemphSeenAny: Bool = false
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
        fmax: Float = 8_000,
        frameStartOffset: Int = 0
    ) {
        self.frameStartOffset = frameStartOffset
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
    // `close()` as the Phase-C reviewer anticipated. Callers (NemotronStreamingEngine)
    // invoke `await mel.close()` on shutdown; if a session is dropped without
    // close, vDSP's setup leaks but the leak is bounded (one allocation per
    // MelExtractor lifetime, typically one per app).
    //
    // `closed` guards against double-destroy if a future caller invokes close()
    // explicitly and then a stray reference triggers another close().
    private var closed = false

    /// Release the underlying vDSP FFT setup. Idempotent. Call once at the end
    /// of the MelExtractor's useful life (e.g., when NemotronStreamingEngine is being
    /// torn down).
    public func close() {
        guard !closed else { return }
        vDSP_destroy_fftsetup(fftSetup)
        closed = true
    }

    // MARK: - Public API

    public func reset() {
        carryoverSamples.removeAll(keepingCapacity: true)
        preemphLastRawSample = 0
        preemphSeenAny = false
        for i in 0..<nMels {
            runningMean[i] = 0
            runningM2[i] = 0
        }
        runningCount = 0
    }

    /// Non-streaming: process the entire buffer, return row-major `[nMels × T]` flattened.
    public func extract(samples: [Float]) -> [Float] {
        // NeMo's pipeline (verified 2026-06-11 against FilterbankFeatures source):
        //   1. Pre-emphasis: x[i] -= 0.97 * x[i-1] (for i >= 1; x[0] unchanged)
        //   2. STFT with center=True, pad_mode='reflect' → input reflect-padded by
        //      n_fft / 2 = 256 samples on each side BEFORE windowing.
        //   3. Power spectrum (mag_power=2.0): |stft|^2
        //   4. Mel filterbank multiply, log(x + 2^-24).
        //
        // We mirror steps 1+2 here so frame timings match torch.stft exactly.
        let prepared = preemphasizeAndPad(samples: samples)
        guard prepared.count >= windowSize else { return [] }

        // With center=True padding, the frame count is ceil(N / hop) for input
        // length N. Equivalent to (N + hop - 1) / hop. Using floor division on
        // the padded length gives the same answer for our hop=160 / win=400 /
        // pad=256 setup.
        let nFrames = (prepared.count - windowSize - frameStartOffset) / hopSize + 1
        var out = [Float](repeating: 0, count: nMels * nFrames)
        var frame = [Float](repeating: 0, count: nMels)

        for t in 0..<nFrames {
            let start = t * hopSize + frameStartOffset
            let end = start + windowSize
            runFrame(samples: prepared, start: start, end: end, into: &frame)
            for m in 0..<nMels {
                out[m * nFrames + t] = frame[m]
            }
        }
        return out
    }

    /// Apply pre-emphasis filter and torch.stft-style reflect padding.
    ///
    /// Pre-emphasis (preemph=0.97): `y[i] = x[i] - 0.97 * x[i-1]`, with `y[0] = x[0]`.
    /// Reflect padding: prepend `x[fftSize/2..1]` reversed and append `x[N-2..N-fftSize/2-1]`
    /// reversed, matching torch.stft(center=True, pad_mode='reflect').
    private func preemphasizeAndPad(samples: [Float]) -> [Float] {
        let n = samples.count
        if n == 0 { return [] }

        // 1. Pre-emphasis (in a fresh buffer to avoid mutating the caller's input).
        var pe = [Float](repeating: 0, count: n)
        pe[0] = samples[0]
        for i in 1..<n {
            pe[i] = samples[i] - 0.97 * samples[i - 1]
        }

        // 2. Reflect padding by fftSize/2 on each side. PyTorch reflect padding
        // skips the boundary sample itself: for input [a, b, c, d], pad 2 on the
        // left → [c, b, a, b, c, d, c, b]. So pad[k] for k in 1..=pad = pe[k].
        let pad = fftSize / 2
        let cappedPad = min(pad, n - 1)  // PyTorch caps reflect pad at len-1
        var padded = [Float]()
        padded.reserveCapacity(n + 2 * cappedPad)
        // Left reflect: pe[pad..1] in reverse order (i.e., pe[pad], pe[pad-1], ..., pe[1])
        for k in stride(from: cappedPad, through: 1, by: -1) {
            padded.append(pe[k])
        }
        padded.append(contentsOf: pe)
        // Right reflect: pe[n-2..n-pad-1]
        for k in stride(from: n - 2, through: max(0, n - cappedPad - 1), by: -1) {
            padded.append(pe[k])
        }
        return padded
    }

    /// Streaming: append `samples` to the carryover, process every full hop available,
    /// return the consumed frames row-major `[nMels × T_consumed]` flattened. Returns
    /// `nil` when no full frame can be produced yet.
    public func extractStreaming(samples: [Float]) -> [Float]? {
        // Apply preemph incrementally across streaming-call boundaries.
        // `preemphLastRawSample` is the raw (non-preemphasized) sample from
        // the previous call; for the very first sample of an utterance we
        // pass it through unchanged (matches NeMo's `y[0] = x[0]` boundary).
        var preemphed = [Float](repeating: 0, count: samples.count)
        for (i, s) in samples.enumerated() {
            if i == 0 && !preemphSeenAny {
                preemphed[i] = s
            } else if i == 0 {
                preemphed[i] = s - 0.97 * preemphLastRawSample
            } else {
                preemphed[i] = s - 0.97 * samples[i - 1]
            }
        }
        if !samples.isEmpty {
            preemphLastRawSample = samples[samples.count - 1]
            preemphSeenAny = true
        }
        carryoverSamples.append(contentsOf: preemphed)
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
                // torch.stft does NOT normalize. Only Apple's vDSP_fft_zrip
                // doubles the amplitude vs the math DFT (4x in power); undo
                // that with /4.
                var invScale: Float = 0.25
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
