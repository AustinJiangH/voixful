// VoixfulTUI — the reusable live-mic session driver.
//
// `runMicSession` captures from the default microphone, feeds any `Transcribing`
// module through a `VoixfulAnalyzer`, and renders the branded terminal UI. It is
// entirely model-agnostic — the only model-specific step (building the
// transcriber) happens in the caller — so every Voixful model backend's CLI can
// share one identical live experience just by handing it a transcriber.

import AVFoundation
import Foundation
import VoixfulSpeech
#if canImport(Darwin)
import Darwin
#endif

/// Error from the live mic session. `CustomStringConvertible` so CLIs print a
/// clean message rather than an enum case.
public struct MicSessionError: Error, CustomStringConvertible {
    public let description: String
    public init(_ description: String) { self.description = description }
}

// SIGINT graceful-stop flag + audio-thread diagnostics. Process-global because a
// C signal handler can't capture context; one mic session runs at a time. Scalar
// stores are atomic on every Darwin arch we target.
nonisolated(unsafe) private var micShouldStop = false
nonisolated(unsafe) private var micBufferCount = 0
nonisolated(unsafe) private var micPeakLevel: Float = 0

private func installMicSignalHandler() {
    let handler: @convention(c) (Int32) -> Void = { _ in micShouldStop = true }
    signal(SIGINT, handler)
}

/// Run a live microphone transcription session until the user presses Ctrl+C.
///
/// - Parameters:
///   - transcriber: any backend's `Transcribing` module (e.g. `NemotronTranscriber`),
///     already configured for streaming with `.volatileResults`.
///   - debug: when `true`, bypasses the full-screen UI so a backend's own
///     per-chunk diagnostics (printed to stderr) stay readable.
public func runMicSession(transcriber: any Transcribing, debug: Bool = false) async throws {
    micShouldStop = false
    let analyzer = VoixfulAnalyzer(modules: [transcriber as any VoixfulSpeechModule])
    let display = MicDisplay(tty: Term.isTTY && !debug)

    // Drive the live UI from the result stream: volatile partials refresh the
    // transcript area; finals commit a line.
    let printer = Task {
        for try await result in transcriber.results {
            let text = String(result.text.characters)
            if result.isFinal {
                await display.commitFinal(text)
            } else {
                await display.setPartial(text)
            }
        }
    }

    // Live audio source. The tap runs on a real-time thread; AsyncStream's
    // continuation.yield is thread-safe and non-blocking, so we push straight
    // from the callback.
    let (inputSequence, inputBuilder) = AsyncStream<AnalyzerInput>.makeStream()

    // Start the analyzer (loads the model) BEFORE installing the tap, so no
    // audio is dropped during cold-start specialization.
    try await analyzer.start(inputSequence: inputSequence)

    let engine = AVAudioEngine()
    let inputNode = engine.inputNode
    let inputFormat = inputNode.inputFormat(forBus: 0)
    guard inputFormat.channelCount > 0, inputFormat.sampleRate > 0 else {
        inputBuilder.finish()
        throw MicSessionError(
            "no microphone configured (channelCount=\(inputFormat.channelCount), "
            + "sampleRate=\(inputFormat.sampleRate))."
        )
    }

    let deviceName = AVCaptureDevice.default(for: .audio)?.localizedName ?? "unknown"
    await display.header(
        device: deviceName,
        sampleRate: Int(inputFormat.sampleRate),
        channels: Int(inputFormat.channelCount)
    )

    micBufferCount = 0
    micPeakLevel = 0
    inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, _ in
        micBufferCount += 1
        micPeakLevel = max(micPeakLevel, peakAmplitude(buffer))
        // The engine reuses `buffer` after this returns, so deep-copy before
        // handing ownership to the analyzer.
        guard let copy = copyPCMBuffer(buffer) else { return }
        inputBuilder.yield(AnalyzerInput(buffer: copy))
    }

    installMicSignalHandler()

    do {
        try engine.start()
    } catch {
        inputBuilder.finish()
        throw MicSessionError("AVAudioEngine.start() failed: \(error.localizedDescription)")
    }

    // Refresh the level meter ~5×/s while we wait for Ctrl+C. The peak is
    // accumulated on the audio thread; we sample-and-reset it here.
    while !micShouldStop {
        try? await Task.sleep(nanoseconds: 200_000_000)
        let level = micPeakLevel
        micPeakLevel = 0
        await display.setLevel(level, hasBuffers: micBufferCount > 0)
    }

    // Teardown. Ctrl+C is a normal stop, not an error — swallow any shutdown
    // error so it never surfaces as an "Error:" message, and ALWAYS reset the
    // terminal (drop the scroll region) so the prompt isn't left broken.
    engine.stop()
    inputNode.removeTap(onBus: 0)
    inputBuilder.finish()
    do {
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        try await printer.value
    } catch {
        printer.cancel()
    }
    await display.finish()

    if micBufferCount == 0 {
        FileHandle.standardError.write(Data(
            "[mic] received no audio — the mic may be muted or held by another app.\n".utf8))
    }
}

/// Peak absolute sample value across ALL channels — a cheap level estimate.
/// Scans every channel because the MacBook mic array can place signal off
/// channel 0.
private func peakAmplitude(_ buffer: AVAudioPCMBuffer) -> Float {
    let n = Int(buffer.frameLength)
    guard n > 0, let channels = buffer.floatChannelData else { return 0 }
    var peak: Float = 0
    for c in 0..<Int(buffer.format.channelCount) {
        let p = channels[c]
        for i in 0..<n { peak = max(peak, abs(p[i])) }
    }
    return peak
}

/// Deep-copy a float PCM buffer so it outlives the tap callback.
private func copyPCMBuffer(_ src: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
    guard let dst = AVAudioPCMBuffer(pcmFormat: src.format, frameCapacity: src.frameLength),
          let s = src.floatChannelData, let d = dst.floatChannelData else { return nil }
    dst.frameLength = src.frameLength
    for ch in 0..<Int(src.format.channelCount) {
        d[ch].update(from: s[ch], count: Int(src.frameLength))
    }
    return dst
}
