// voixful — brand CLI built on the SpeechAnalyzer-shaped Voixful API.
//
// Dogfoods the new surface: `transcribe` drives `VoixfulAnalyzer` +
// `NemotronTranscriber` exactly the way a third-party app would.
//
//   voixful transcribe input.wav --model path/to.aimodel --output out.txt
//   voixful languages

import ArgumentParser
import AVFoundation
import Foundation
import VoixfulSpeech
import VoixfulTUI
import VoixfulNemotron
import NemotronASR

@main
struct Voixful: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "voixful",
        abstract: "Voixful on-device speech — a SpeechAnalyzer-shaped API over local models.",
        subcommands: [Transcribe.self, Mic.self, Parakeet.self, Cohere.self, Languages.self]
    )
}

// MARK: - cohere (attention encoder-decoder backend)

struct Cohere: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Transcribe a file with Cohere Transcribe 2B (attention encoder-decoder)."
    )

    @Argument(help: "Input audio file (WAV/CAF/AIFF/… — any rate/format).")
    var input: String

    @Option(help: "Path to the Cohere .aimodel directory.")
    var model: String

    @Option(help: "Path to the Cohere SentencePiece vocab (one piece per line).")
    var pieces: String

    @Option(help: "Write the transcript here (default: stdout).")
    var output: String?

    func run() async throws {
        let samples = try loadMono16k(path: input)
        let engine: CohereEngine
        do {
            engine = try await CohereEngine(
                modelURL: URL(fileURLWithPath: model),
                piecesURL: URL(fileURLWithPath: pieces)
            )
        } catch {
            throw ValidationError("could not load Cohere engine: \(error)")
        }
        let text = try await engine.transcribe(audio: samples)
        if let output {
            try text.write(toFile: output, atomically: true, encoding: .utf8)
        } else {
            print(text)
        }
    }
}

// MARK: - parakeet (full-context TDT backend — FluidAudio comparison)

struct Parakeet: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Transcribe a file with the full-context Parakeet TDT 0.6B model."
    )

    @Argument(help: "Input audio file (WAV/CAF/AIFF/… — any rate/format).")
    var input: String

    @Option(help: "Path to the Parakeet .aimodel directory.")
    var model: String

    @Option(help: "Path to the Parakeet SentencePiece vocab (one piece per line).")
    var pieces: String

    @Option(help: "Write the transcript here (default: stdout).")
    var output: String?

    func run() async throws {
        let samples = try loadMono16k(path: input)

        let engine: ParakeetEngine
        do {
            engine = try await ParakeetEngine(
                modelURL: URL(fileURLWithPath: model),
                piecesURL: URL(fileURLWithPath: pieces)
            )
        } catch {
            throw ValidationError("could not load Parakeet engine: \(error)")
        }

        let text = try await engine.transcribe(audio: samples)
        if let output {
            try text.write(toFile: output, atomically: true, encoding: .utf8)
        } else {
            print(text)
        }
    }
}

/// Decode any audio file to 16 kHz mono Float samples for the engine.
/// Shared by the `parakeet` and `cohere` subcommands.
private func loadMono16k(path: String) throws -> [Float] {
        let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
        guard let target = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        ) else {
            throw ValidationError("could not build 16 kHz mono format")
        }
        guard let converter = AVAudioConverter(from: file.processingFormat, to: target) else {
            throw ValidationError("could not build audio converter for \(path)")
        }

        guard let inBuf = AVAudioPCMBuffer(
            pcmFormat: file.processingFormat,
            frameCapacity: AVAudioFrameCount(file.length)
        ) else {
            throw ValidationError("could not allocate input buffer for \(path)")
        }
        try file.read(into: inBuf)

        let ratio = target.sampleRate / file.processingFormat.sampleRate
        let outCapacity = AVAudioFrameCount(Double(inBuf.frameLength) * ratio) + 1024
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: outCapacity) else {
            throw ValidationError("could not allocate output buffer for \(path)")
        }

        var fed = false
        var error: NSError?
        let status = converter.convert(to: outBuf, error: &error) { _, inputStatus in
            if fed {
                inputStatus.pointee = .noDataNow
                return nil
            }
            fed = true
            inputStatus.pointee = .haveData
            return inBuf
        }
        if let error { throw ValidationError("resample failed: \(error)") }
        if status == .error { throw ValidationError("resample returned .error") }

        let n = Int(outBuf.frameLength)
        guard let ch = outBuf.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: ch[0], count: n))
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
        // Build the model-specific transcriber in streaming mode with volatile
        // reporting (fixed cost per chunk, live partials), then hand it to the
        // shared, model-agnostic mic session driver, which captures audio,
        // drives the analyzer, and renders the live UI. A new model's CLI does
        // exactly this with its own transcriber.
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

        try await runMicSession(transcriber: transcriber, debug: debug)
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
