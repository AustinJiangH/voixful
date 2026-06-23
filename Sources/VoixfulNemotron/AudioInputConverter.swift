// VoixfulNemotron — AVAudioPCMBuffer → [Float] 16 kHz mono.
//
// Centralizes the format conversion that previously lived (twice) in the CLI
// and the SwiftUI demo. Used only inside `NemotronTranscriber`'s actor
// isolation, so it carries mutable converter state without needing Sendable.

import AVFoundation
import VoixfulSpeech

final class AudioInputConverter {
    /// The format `StreamingASR.feed`/`transcribe` requires.
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
    func convert(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
        let inFormat = buffer.format

        if inFormat.sampleRate == 16_000,
           inFormat.channelCount == 1,
           inFormat.commonFormat == .pcmFormatFloat32 {
            return Self.floatChannel(buffer)
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
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio + 256.0)
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
            return buffer
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
}
