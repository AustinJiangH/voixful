// VoixfulSpeech — AnalyzerInput.
//
// Mirrors Apple's `AnalyzerInput`: a unit of audio fed to the analyzer. Accepts
// any `AVAudioPCMBuffer` (any sample rate / channel layout); each backend
// converts to its required format internally.

import AVFoundation
import CoreMedia

/// One buffer of audio handed to `VoixfulAnalyzer`.
///
/// `@unchecked Sendable`: `AVAudioPCMBuffer` is a non-`Sendable` reference type,
/// but the contract is that ownership is transferred to the analyzer on
/// construction and the caller does not mutate the buffer afterward — the same
/// hand-off discipline used throughout AVFoundation audio pipelines.
public struct AnalyzerInput: @unchecked Sendable {
    /// The audio samples. Any rate/format; the backend resamples as needed.
    public let buffer: AVAudioPCMBuffer

    /// Optional presentation time of the first sample on the session timeline.
    /// Seeds the `CMTimeRange` on emitted `TranscriptionResult`s. `nil` ⇒ 0.
    public let bufferStartTime: CMTime?

    public init(buffer: AVAudioPCMBuffer, bufferStartTime: CMTime? = nil) {
        self.buffer = buffer
        self.bufferStartTime = bufferStartTime
    }
}
