// VoixfulSpeech — AVAudioPCMBuffer → [Float] 16 kHz mono.
//
// Centralizes the format conversion that previously lived (twice) in the CLI
// and the SwiftUI demo. Used only inside `NemotronStreamingTranscriber`'s actor
// isolation, so it carries mutable converter state without needing Sendable.

import AVFoundation


public final class AudioInputConverter {
    /// The format `NemotronStreamingEngine.feed`/`transcribe` requires.
    private static let target = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 16_000,
        channels: 1,
        interleaved: false
    )!

    private var converter: AVAudioConverter?
    private var sourceFormat: AVAudioFormat?

    /// Convert one buffer to 16 kHz mono `Float`. Fast-paths buffers that are
    /// already in the target format; otherwise builds (and caches) an
    /// `AVAudioConverter` for the incoming format.
    public init() {}

    public func convert(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
        // CRITICAL: down-mix multichannel input to mono BEFORE the rate
        // converter. `AVAudioConverter` cannot mix a *discrete* N-channel layout
        // (e.g. the MacBook Pro mic array's 3-channel discrete format) down to
        // mono — it returns `.haveData` but emits all-zero (silent) output. We
        // average the channels into a mono buffer ourselves first, then let the
        // cached converter handle only the sample-rate change (mono → 16 kHz).
        // Stereo/mono inputs pass through this unchanged in behavior.
        let source = Self.downmixToMono(buffer)
        let inFormat = source.format

        if inFormat.sampleRate == 16_000,
           inFormat.channelCount == 1,
           inFormat.commonFormat == .pcmFormatFloat32 {
            return Self.floatChannel(source)
        }

        if converter == nil || !(sourceFormat?.isEqual(inFormat) ?? false) {
            guard let made = AVAudioConverter(from: inFormat, to: Self.target) else {
                throw VoixfulError.audioConversionFailed(
                    "cannot build AVAudioConverter \(inFormat) → 16 kHz mono Float32"
                )
            }
            converter = made
            sourceFormat = inFormat
        }
        guard let converter else {
            throw VoixfulError.audioConversionFailed("converter unavailable")
        }

        let ratio = 16_000.0 / inFormat.sampleRate
        let capacity = AVAudioFrameCount(Double(source.frameLength) * ratio + 256.0)
        guard let output = AVAudioPCMBuffer(pcmFormat: Self.target, frameCapacity: capacity) else {
            throw VoixfulError.audioConversionFailed("cannot allocate output buffer")
        }

        var convError: NSError?
        var consumed = false
        let status = converter.convert(to: output, error: &convError) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return source
        }
        if status == .error || convError != nil {
            throw VoixfulError.audioConversionFailed(
                "AVAudioConverter failed: \(convError?.localizedDescription ?? "unknown")"
            )
        }
        return Self.floatChannel(output)
    }

    private static func floatChannel(_ buffer: AVAudioPCMBuffer) -> [Float] {
        let n = Int(buffer.frameLength)
        guard n > 0, let channels = buffer.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: channels[0], count: n))
    }

    /// Average all channels of a non-interleaved Float32 buffer into a single
    /// mono buffer at the SAME sample rate. Returns `buffer` unchanged when it
    /// is already mono (or not float/non-interleaved, in which case we leave the
    /// down-mix to the rate converter). This is the down-mix `AVAudioConverter`
    /// fails to perform for discrete multichannel layouts.
    private static func downmixToMono(_ buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer {
        let inFormat = buffer.format
        let channelCount = Int(inFormat.channelCount)
        guard channelCount > 1,
              inFormat.commonFormat == .pcmFormatFloat32,
              !inFormat.isInterleaved,
              let src = buffer.floatChannelData else {
            return buffer
        }

        guard let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: inFormat.sampleRate,
            channels: 1,
            interleaved: false
        ), let mono = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: buffer.frameLength),
           let dst = mono.floatChannelData else {
            return buffer
        }

        let n = Int(buffer.frameLength)
        mono.frameLength = buffer.frameLength
        let scale = 1.0 / Float(channelCount)
        let out = dst[0]
        for i in 0..<n {
            var sum: Float = 0
            for c in 0..<channelCount { sum += src[c][i] }
            out[i] = sum * scale
        }
        return mono
    }
}
