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

    @Flag(help: "Print per-chunk feed diagnostics (sample count, peak, partial text) to stderr.")
    var debug: Bool = false

    func run() async throws {
        // Build the brand surface with volatile reporting so we get live
        // partials in addition to the final result. Streaming mode drives
        // StreamingASR.feed() incrementally — fixed cost per chunk, no
        // re-transcription — so it stays fast no matter how long the session
        // runs. Let AVAudioEngine surface the mic prompt implicitly (the
        // embedded Info.plist gives it a usage string).
        let transcriber: NemotronTranscriber
        do {
            transcriber = try NemotronTranscriber(
                modelURL: URL(fileURLWithPath: model),
                locale: Locale(identifier: language),
                chunkSizeMS: chunk,
                mode: .streaming,
                reportingOptions: [.volatileResults],
                attributeOptions: [],
                debug: debug
            )
        } catch {
            throw ValidationError("could not create transcriber: \(error)")
        }
        let analyzer = VoixfulAnalyzer(modules: [transcriber])

        let display = MicDisplay(tty: Term.isTTY)

        // Drive the live UI from the result stream: volatile partials refresh
        // the sticky status line; finals commit into the scrollback above it.
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

        // Branded header banner (also surfaces the device so a flat level can be
        // traced to the wrong/muted input).
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
            throw ValidationError("AVAudioEngine.start() failed: \(error.localizedDescription)")
        }

        // Refresh the level meter ~5×/s while we wait for Ctrl+C. The peak is
        // accumulated on the audio thread; we sample-and-reset it here. The
        // meter and the partial share ONE sticky line owned by `display`, so
        // they never fight over the terminal the way the old stdout/stderr
        // split did.
        while !micShouldStop {
            try? await Task.sleep(nanoseconds: 200_000_000)
            let level = micPeakLevel
            micPeakLevel = 0
            await display.setLevel(level, hasBuffers: micBufferCount > 0)
        }

        // Teardown: stop capture, close the input stream, finalize → final
        // result, wait for the printer to flush it, then restore the terminal.
        // Ctrl+C is a normal stop, not an error — swallow any shutdown error so
        // it never surfaces as an "Error:" message, and ALWAYS reset the
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

// MARK: - Terminal styling

/// Tiny ANSI + terminal helpers for the live mic UI.
private enum Term {
    /// Only style when stdout is a real terminal; piped output stays plain.
    static let isTTY = isatty(STDOUT_FILENO) != 0

    /// Current terminal width (columns), best-effort.
    static var cols: Int {
        var ws = winsize()
        if ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0, ws.ws_col > 0 { return Int(ws.ws_col) }
        if let c = ProcessInfo.processInfo.environment["COLUMNS"], let n = Int(c), n > 0 { return n }
        return 80
    }

    /// Current terminal height (rows), best-effort.
    static var rows: Int {
        var ws = winsize()
        if ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0, ws.ws_row > 0 { return Int(ws.ws_row) }
        if let r = ProcessInfo.processInfo.environment["LINES"], let n = Int(r), n > 0 { return n }
        return 24
    }

    static let reset = "\u{1B}[0m"
    static let bold = "\u{1B}[1m"
    static let dim = "\u{1B}[2m"
    static let italic = "\u{1B}[3m"
    static func fg(_ c: Int) -> String { "\u{1B}[38;5;\(c)m" }

    /// "voixful" rendered as a teal→green 256-color gradient. Visible width 7.
    static func brand() -> String {
        let palette = [45, 44, 43, 79, 115, 151, 187]
        var s = bold
        for (i, ch) in Array("voixful").enumerated() { s += fg(palette[i]) + String(ch) }
        return s + reset
    }
}

// MARK: - Live mic display

/// Owns the terminal during `voixful mic`. The transcription flows and wraps in
/// the main scroll area (so the whole thing is visible / scrollable), while a
/// fixed two-line footer is pinned at the bottom via a DEC scroll region:
///   line 1: voixful brand + colored level meter + numeric peak
///   line 2: input device · rate · channels   ·   Ctrl+C to stop
/// All writes funnel through this actor, so the footer and the transcript never
/// fight over the cursor. Falls back to plain line output when stdout is not a
/// terminal or is too short for a pinned footer.
private actor MicDisplay {
    private let tty: Bool
    private var rows = 0
    private var enabled = false          // pinned-footer mode active
    private var device = ""
    private var sr = 0
    private var ch = 0
    private var printed = ""             // transcript text already emitted
    private var peak: Float = 0
    private var hasBuffers = true

    private static let footerHeight = 2

    init(tty: Bool) { self.tty = tty }

    private func out(_ s: String) { FileHandle.standardOutput.write(Data(s.utf8)) }

    func header(device: String, sampleRate: Int, channels: Int) {
        self.device = device
        self.sr = sampleRate
        self.ch = channels
        rows = Term.rows
        guard tty, rows >= 6 else {
            out("voixful · listening — speak (Ctrl+C to stop)\n")
            return
        }
        enabled = true
        out("\u{1B}[2J\u{1B}[H")                              // clear screen, home
        out("\u{1B}[1;\(rows - Self.footerHeight)r")          // scroll region above footer
        out("\u{1B}[1;1H")                                    // cursor to top of region
        out("\(Term.brand())  \(Term.dim)· on-device speech — speak\(Term.reset)\n\n")
        printed = ""
        drawFooter()
    }

    func setPartial(_ text: String) {
        append(text.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    func setLevel(_ p: Float, hasBuffers: Bool) {
        peak = p
        self.hasBuffers = hasBuffers
        if enabled { drawFooter() }
    }

    func commitFinal(_ text: String) {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty else { return }
        if enabled {
            append(t)
            out("\n")            // seal the final line; next utterance starts fresh
            printed = ""
            drawFooter()
        } else {
            out(t + "\n")
        }
    }

    /// Reset the terminal: drop the scroll region and move below the footer so
    /// the shell prompt lands cleanly. The transcript stays on screen.
    func finish() {
        guard enabled else { return }
        out("\u{1B}[r")                       // reset scroll region (full screen)
        out("\u{1B}[\(rows);1H\n")            // below the footer
        enabled = false
    }

    /// Append the cumulative transcript's new suffix into the scroll area. The
    /// streaming text is append-only (greedy RNN-T commits tokens), so we print
    /// only the delta; on the rare revision we start a fresh line.
    private func append(_ text: String) {
        guard enabled, !text.isEmpty else { return }
        let delta: String
        if text.hasPrefix(printed) {
            delta = String(text.dropFirst(printed.count))
        } else {
            delta = "\n" + text
        }
        printed = text
        guard !delta.isEmpty else { return }
        out(delta)               // flows + wraps + scrolls within the region
        drawFooter()             // a scroll may have occurred; keep footer fresh
    }

    private func drawFooter() {
        guard enabled else { return }
        out("\u{1B}7")                                                  // save cursor
        out("\u{1B}[\(rows - 1);1H\u{1B}[2K\(footerStatus())")          // line 1
        out("\u{1B}[\(rows);1H\u{1B}[2K\(footerInfo())")                // line 2
        out("\u{1B}8")                                                  // restore cursor
    }

    private func footerStatus() -> String {
        let width = 18
        let level = min(max(peak, 0), 1)
        let filled = Int((level * Float(width)).rounded())
        let color = peak > 0.6 ? 203 : (peak > 0.22 ? 215 : 78)
        let meter = Term.fg(color) + String(repeating: "▮", count: filled)
            + Term.dim + String(repeating: "·", count: width - filled) + Term.reset
        let peakStr = String(format: "%.2f", peak)
        let warn = hasBuffers ? "" : "   \(Term.fg(203))no audio — check mic permission\(Term.reset)"
        return " \(Term.brand())  \(Term.fg(240))▕\(Term.reset)\(meter)\(Term.fg(240))▏\(Term.reset)"
            + " \(Term.dim)\(peakStr)\(Term.reset)\(warn)"
    }

    private func footerInfo() -> String {
        let dev = device.count > 34 ? String(device.prefix(33)) + "…" : device
        return " \(Term.dim)mic\(Term.reset) \(dev) · \(sr / 1000) kHz · \(ch)ch"
            + "   \(Term.fg(240))·\(Term.reset)   \(Term.dim)Ctrl+C to stop\(Term.reset)"
    }
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
