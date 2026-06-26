// voixful — brand CLI built on the SpeechAnalyzer-shaped Voixful API.
//
// Dogfoods the new surface: `transcribe` drives `VoixfulAnalyzer` +
// `NemotronStreamingTranscriber` exactly the way a third-party app would.
//
//   voixful transcribe input.wav --model path/to.aimodel --output out.txt
//   voixful languages

import ArgumentParser
import AVFoundation
import Foundation
import VoixfulSpeech
import VoixfulTUI
import VoixfulNemotronStreaming
import VoixfulParakeetTDT
import VoixfulCohereTranscribe
import VoixfulFullContext
import VoixfulEngine

@main
struct Voixful: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "voixful",
        abstract: "Voixful on-device speech — a SpeechAnalyzer-shaped API over local models.",
        subcommands: [Transcribe.self, Mic.self, Languages.self]
    )
}

// MARK: - transcribe

// `ModelBackend` (VoixfulEngine) is the single source of truth for the backend
// set + the detection heuristic; expose it as a CLI option value here.
extension ModelBackend: ExpressibleByArgument {}

struct Transcribe: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Transcribe an audio file to text."
    )

    @Argument(help: "Input audio file (WAV/CAF/AIFF/… — any rate/format).")
    var input: String

    @Option(help: "Path to the .aimodel directory.")
    var model: String

    @Option(help: "Force the model architecture (nemotron/parakeet/cohere). Default: auto-detect from the model.")
    var backend: ModelBackend?

    @Option(help: "SentencePiece vocab for parakeet/cohere. Default: sibling 'pieces.txt' of the model.")
    var pieces: String?

    @Option(help: "Write the transcript here (default: stdout).")
    var output: String?

    @Option(name: .shortAndLong, help: "BCP-47 locale, e.g. en-US.")
    var language: String = "en-US"

    @Option(help: "Nemotron streaming chunk size in ms (80, 160, 320, 560, 1120).")
    var chunk: Int = 320

    @Option(help: "Output format: text or json.")
    var format: String = "text"

    func run() async throws {
        // Build the brand surface — every backend is a `Transcribing` module
        // driven through the same `VoixfulAnalyzer` (see README.md).
        let modelURL = URL(fileURLWithPath: model)
        let locale = Locale(identifier: language)

        // Read the model's functions once: drives both backend auto-detection
        // (unless --backend forces it) and the Nemotron batch-vs-streaming mode.
        let fns = try ModelBackend.assetFunctions(modelURL: modelURL)
        let resolved = backend ?? ModelBackend.classify(fns)

        let transcriber: any Transcribing
        do {
            switch resolved {
            case .nemotron:
                // Batch needs the baked `preprocess_audio`; without it (every
                // convert-pipeline model), stream. No user flag required.
                transcriber = try NemotronStreamingTranscriber(
                    modelURL: modelURL, locale: locale, chunkSizeMS: chunk,
                    mode: fns.names.contains("preprocess_audio") ? .batch : .streaming,
                    reportingOptions: [], attributeOptions: [.audioTimeRange])
            case .parakeet:
                transcriber = ParakeetTDTTranscriber(
                    modelURL: modelURL, piecesURL: piecesURL(modelURL), locale: locale)
            case .cohere:
                transcriber = CohereTranscriber(
                    modelURL: modelURL, piecesURL: piecesURL(modelURL), locale: locale)
            }
        } catch {
            throw ValidationError("could not create \(resolved.rawValue) transcriber: \(error)")
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

    /// `--pieces` if given, else `pieces.txt` bundled inside the `.aimodel` dir
    /// (self-contained model). Parakeet/cohere only; Nemotron resolves its own.
    private func piecesURL(_ modelURL: URL) -> URL {
        if let pieces { return URL(fileURLWithPath: pieces) }
        return modelURL.appendingPathComponent("pieces.txt")
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
        let transcriber: NemotronStreamingTranscriber
        do {
            transcriber = try NemotronStreamingTranscriber(
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
        let tags = NemotronStreamingTranscriber.supportedLocales
            .map { $0.identifier(.bcp47) }
            .sorted()
            .joined(separator: "\n")
        print(tags)
    }
}
