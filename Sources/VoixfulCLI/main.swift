// voixful — brand CLI built on the SpeechAnalyzer-shaped Voixful API.
//
// Dogfoods the new surface: `transcribe` drives `VoixfulAnalyzer` +
// `NemotronTranscriber` exactly the way a third-party app would.
//
//   voixful transcribe input.wav --model path/to.aimodel --output out.txt
//   voixful languages

import ArgumentParser
import AVFoundation
import Darwin
import Foundation
import VoixfulSpeech
import VoixfulNemotron

@main
struct Voixful: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "voixful",
        abstract: "Voixful on-device speech — a SpeechAnalyzer-shaped API over local models.",
        subcommands: [Transcribe.self, Mic.self, Languages.self]
    )
}

// MARK: - transcribe

struct Transcribe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Transcribe an audio file to text."
    )

    @Argument(help: "Input audio file (WAV/CAF/AIFF/… — any rate/format).")
    var input: String

    @Option(help: "Path to the .aimodel directory.")
    var model: String

    @Option(help: "Write the transcript here (default: stdout).")
    var output: String?

    @Option(name: .shortAndLong, help: "BCP-47 locale, e.g. en-US.")
    var language: String = "en-US"

    @Option(help: "Streaming chunk size in ms (80, 160, 320, 560, 1120).")
    var chunk: Int = 320

    @Option(help: "Output format: text or json.")
    var format: String = "text"

    @Flag(help: "Use the incremental streaming feed() path instead of batch transcribe.")
    var streaming: Bool = false

    func run() async throws {
        // Build the brand surface — identical in shape to Apple's SpeechAnalyzer.
        let transcriber: NemotronTranscriber
        do {
            transcriber = try NemotronTranscriber(
                modelURL: URL(fileURLWithPath: model),
                locale: Locale(identifier: language),
                chunkSizeMS: chunk,
                mode: streaming ? .streaming : .batch,
                reportingOptions: [],
                attributeOptions: [.audioTimeRange]
            )
        } catch {
            throw ValidationError("could not create transcriber: \(error)")
        }
        let analyzer = VoixfulAnalyzer(modules: [transcriber])

        // Load the whole file into one PCM buffer; the transcriber resamples.
        let buffer = try loadBuffer(path: input)

        // Collect results off the transcriber's stream while we feed audio.
        let collector = Task { () throws -> TranscriptionResult? in
            var last: TranscriptionResult?
            for try await result in transcriber.results where result.isFinal {
                last = result
            }
            return last
        }

        let (inputSequence, inputBuilder) = AsyncStream<AnalyzerInput>.makeStream()
        try await analyzer.start(inputSequence: inputSequence)
        inputBuilder.yield(AnalyzerInput(buffer: buffer, bufferStartTime: .zero))
        inputBuilder.finish()
        try await analyzer.finalizeAndFinishThroughEndOfInput()

        guard let result = try await collector.value else {
            throw ValidationError("no transcription produced")
        }

        let rendered = render(result, format: format, language: language)
        if let output {
            try rendered.write(toFile: output, atomically: true, encoding: .utf8)
        } else {
            print(rendered)
        }
    }

    /// Load an audio file into a single PCM buffer at its native format.
    private func loadBuffer(path: String) throws -> AVAudioPCMBuffer {
        let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
        let format = file.processingFormat
        guard let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: AVAudioFrameCount(file.length)
        ) else {
            throw ValidationError("could not allocate buffer for \(path)")
        }
        try file.read(into: buffer)
        return buffer
    }

    /// Render a result as plain text or JSON.
    private func render(_ result: TranscriptionResult, format: String, language: String) -> String {
        let text = String(result.text.characters)
        switch format.lowercased() {
        case "json":
            let payload = TranscriptJSON(
                text: text,
                language: language,
                start: result.range.start.seconds,
                duration: result.range.duration.seconds
            )
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            if let data = try? encoder.encode(payload), let s = String(data: data, encoding: .utf8) {
                return s
            }
            return text
        default:
            return text
        }
    }
}

private struct TranscriptJSON: Encodable {
    let text: String
    let language: String
    let start: Double
    let duration: Double
}

// MARK: - mic

/// Set by the SIGINT handler so the async mic loop can stop gracefully. A C
/// signal handler can't capture context, so we stash a global flag the loop
/// polls. Bool stores are atomic on every Darwin arch we target.
nonisolated(unsafe) private var micShouldStop = false

/// Cheap diagnostics updated from the audio tap and read by the idle loop, so
/// "nothing shows up" can be diagnosed as silence (mic muted / wrong device)
/// vs. no audio at all (no permission / no buffers). Scalar stores are atomic
/// on every Darwin arch we target; exact values don't matter for a meter.
nonisolated(unsafe) private var micBufferCount = 0
nonisolated(unsafe) private var micPeakLevel: Float = 0

private func installMicSignalHandler() {
    let handler: @convention(c) (Int32) -> Void = { _ in micShouldStop = true }
    signal(SIGINT, handler)
}

/// Ensure the process is authorized to capture audio. For an un-bundled CLI the
/// implicit TCC prompt is unreliable; requesting access explicitly triggers it
/// and lets us fail with a clear message instead of silently receiving silence.
private func ensureMicrophoneAccess() async throws {
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized:
        return
    case .notDetermined:
        guard await AVCaptureDevice.requestAccess(for: .audio) else {
            throw ValidationError("microphone access was denied at the prompt.")
        }
    case .denied, .restricted:
        throw ValidationError(
            "microphone access is off for this binary. Enable it in "
            + "System Settings → Privacy & Security → Microphone, then re-run."
        )
    @unknown default:
        return
    }
}

struct Mic: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Transcribe live from the default microphone (Ctrl+C to stop)."
    )

    @Option(help: "Path to the .aimodel directory.")
    var model: String

    @Option(name: .shortAndLong, help: "BCP-47 locale, e.g. en-US.")
    var language: String = "en-US"

    @Option(help: "Streaming chunk size in ms (80, 160, 320, 560, 1120).")
    var chunk: Int = 320

    func run() async throws {
        // Gate on microphone access first — otherwise an un-bundled CLI can
        // start the engine but receive only silence, and nothing ever prints.
        try await ensureMicrophoneAccess()

        // Build the brand surface with volatile reporting so we get live
        // partials in addition to the final result.
        let transcriber: NemotronTranscriber
        do {
            transcriber = try NemotronTranscriber(
                modelURL: URL(fileURLWithPath: model),
                locale: Locale(identifier: language),
                chunkSizeMS: chunk,
                mode: .streaming,
                reportingOptions: [.volatileResults],
                attributeOptions: []
            )
        } catch {
            throw ValidationError("could not create transcriber: \(error)")
        }
        let analyzer = VoixfulAnalyzer(modules: [transcriber])

        // Print results as they arrive: volatile partials update one line in
        // place; finals land on their own line.
        let printer = Task {
            for try await result in transcriber.results {
                let text = String(result.text.characters)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty else { continue }
                let line = result.isFinal ? "\u{1B}[2K\r\(text)\n" : "\u{1B}[2K\r\(truncateTail(text))"
                FileHandle.standardOutput.write(Data(line.utf8))
            }
        }

        // Live audio source. The tap runs on a real-time thread; AsyncStream's
        // continuation.yield is thread-safe and non-blocking, so we push input
        // straight from the callback.
        let (inputSequence, inputBuilder) = AsyncStream<AnalyzerInput>.makeStream()

        // Start the analyzer (loads the model) BEFORE we install the tap, so no
        // audio is dropped during the cold-start specialization.
        try await analyzer.start(inputSequence: inputSequence)

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let inputFormat = inputNode.inputFormat(forBus: 0)
        guard inputFormat.channelCount > 0, inputFormat.sampleRate > 0 else {
            inputBuilder.finish()
            throw ValidationError(
                "no microphone configured (channelCount=\(inputFormat.channelCount), "
                + "sampleRate=\(inputFormat.sampleRate))."
            )
        }

        // Report exactly which device + format we're capturing, so a flat
        // level (silence) can be traced to the wrong/muted input device.
        let deviceName = AVCaptureDevice.default(for: .audio)?.localizedName ?? "unknown"
        let info = "[mic] input: \(deviceName) — \(Int(inputFormat.sampleRate)) Hz, "
            + "\(inputFormat.channelCount) ch\n"
        FileHandle.standardError.write(Data(info.utf8))

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
        FileHandle.standardError.write(Data("[mic] Listening — speak; Ctrl+C to stop.\n".utf8))

        do {
            try engine.start()
        } catch {
            inputBuilder.finish()
            throw ValidationError("AVAudioEngine.start() failed: \(error.localizedDescription)")
        }

        // Idle until Ctrl+C, printing a 1 Hz input-level meter to stderr so a
        // silent mic (no level) is distinguishable from no input at all
        // (no buffers).
        var ticks = 0
        while !micShouldStop {
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            ticks += 1
            let level = micPeakLevel
            micPeakLevel = 0
            let bars = Int((min(level, 1.0) * 20).rounded())
            let meter = String(repeating: "▮", count: bars) + String(repeating: "·", count: 20 - bars)
            let status = micBufferCount == 0
                ? "no audio buffers — check input device/permission"
                : "level [\(meter)]"
            FileHandle.standardError.write(Data("\u{1B}[2K\r[mic] \(status)".utf8))
            _ = ticks
        }
        FileHandle.standardError.write(Data("\u{1B}[2K\r".utf8))

        // Teardown: stop capture, close the input stream, finalize → final
        // result, then wait for the printer to flush it.
        engine.stop()
        inputNode.removeTap(onBus: 0)
        inputBuilder.finish()
        try await analyzer.finalizeAndFinishThroughEndOfInput()
        try await printer.value

        if micBufferCount == 0 {
            FileHandle.standardError.write(Data(
                "[mic] received no audio. The mic may be muted or another app holds it.\n".utf8))
        }
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

/// Keep a partial line within the terminal width by trimming from the left so
/// the most recently spoken words stay visible.
private func truncateTail(_ text: String) -> String {
    let cols = Int(ProcessInfo.processInfo.environment["COLUMNS"] ?? "") ?? 100
    guard text.count > cols - 2 else { return text }
    return "…" + String(text.suffix(cols - 3))
}

// MARK: - languages

struct Languages: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "List locales the bundled model supports."
    )

    func run() async throws {
        let tags = NemotronTranscriber.supportedLocales
            .map { $0.identifier(.bcp47) }
            .sorted()
            .joined(separator: "\n")
        print(tags)
    }
}
