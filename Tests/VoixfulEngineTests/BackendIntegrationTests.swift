// End-to-end backend tests across all three model architectures (Nemotron
// streaming RNN-T, Parakeet full-context TDT, Cohere AED).
//
// These need the real `.aimodel` assets, which are large and not in the repo, so
// they are GATED on the `VOIXFUL_TEST_MODELS_DIR` environment variable and skip
// otherwise (a normal `swift test` skips them). Run with:
//
//   VOIXFUL_TEST_MODELS_DIR=/path/to/models swift test --filter BackendIntegration
//
// The dir is expected to contain the palette4 variants, each self-contained
// (pieces.txt bundled inside the .aimodel dir):
//   nemotron-speech-streaming-en-0.6b-palette4.aimodel
//   parakeet-tdt-0.6b-v3-palette4.aimodel
//   cohere-transcribe-03-2026-palette4.aimodel

import AVFoundation
import XCTest
@testable import VoixfulEngine

final class BackendIntegrationTests: XCTestCase {

    // Verbose asset names (full HF model id + precision); each is self-contained
    // with a bundled pieces.txt.
    static let nemotron = "nemotron-speech-streaming-en-0.6b-palette4.aimodel"
    static let parakeet = "parakeet-tdt-0.6b-v3-palette4.aimodel"
    static let cohere = "cohere-transcribe-03-2026-palette4.aimodel"

    private func modelsDir() throws -> URL {
        guard let d = ProcessInfo.processInfo.environment["VOIXFUL_TEST_MODELS_DIR"], !d.isEmpty else {
            throw XCTSkip("set VOIXFUL_TEST_MODELS_DIR to run the backend integration tests")
        }
        return URL(fileURLWithPath: d)
    }

    private func require(_ url: URL) throws {
        if !FileManager.default.fileExists(atPath: url.path) {
            throw XCTSkip("asset not present: \(url.lastPathComponent)")
        }
    }

    /// `fixtures/sample.wav` decoded to 16 kHz mono Float (the same clip the CLI uses).
    private func sampleAudio() throws -> [Float] {
        // #filePath: Tests/VoixfulEngineTests/<this>.swift -> repo root/fixtures/sample.wav
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let wav = root.appendingPathComponent("fixtures/sample.wav")
        try require(wav)
        let file = try AVAudioFile(forReading: wav)
        guard
            let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
                                    channels: 1, interleaved: false),
            let conv = AVAudioConverter(from: file.processingFormat, to: fmt),
            let inBuf = AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                         frameCapacity: AVAudioFrameCount(file.length))
        else { throw XCTSkip("could not set up audio decode") }
        try file.read(into: inBuf)
        let cap = AVAudioFrameCount(Double(inBuf.frameLength) * 16_000
            / file.processingFormat.sampleRate) + 1024
        guard let outBuf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: cap) else {
            throw XCTSkip("could not allocate output buffer")
        }
        var fed = false
        var err: NSError?
        _ = conv.convert(to: outBuf, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true; status.pointee = .haveData; return inBuf
        }
        if let err { throw err }
        let n = Int(outBuf.frameLength)
        guard let ch = outBuf.floatChannelData else { return [] }
        return Array(UnsafeBufferPointer(start: ch[0], count: n))
    }

    func testBackendAutoDetection() throws {
        let dir = try modelsDir()
        for (file, expected): (String, ModelBackend) in [
            (Self.nemotron, .nemotron),
            (Self.parakeet, .parakeet),
            (Self.cohere, .cohere),
        ] {
            let url = dir.appendingPathComponent(file)
            try require(url)
            XCTAssertEqual(try ModelBackend.detect(modelURL: url), expected,
                           "auto-detect for \(file)")
        }
    }

    /// A wrong-size vocab must fail loudly (vocab mismatch), not detokenize to
    /// garbage — load the Parakeet model with Nemotron's 1024-piece vocab.
    func testWrongVocabRejected() async throws {
        let dir = try modelsDir()
        let model = dir.appendingPathComponent(Self.parakeet)
        // Nemotron's 1024-piece vocab, bundled inside the Nemotron asset.
        let wrongPieces = dir.appendingPathComponent(Self.nemotron)
            .appendingPathComponent("pieces.txt")
        try require(model)
        try require(wrongPieces)
        do {
            _ = try await ParakeetTDTEngine(modelURL: model, piecesURL: wrongPieces)
            XCTFail("expected a vocab-mismatch error")
        } catch {
            XCTAssertTrue("\(error)".contains("vocab mismatch"), "got: \(error)")
        }
    }

    func testNemotronStreamingTranscribes() async throws {
        let dir = try modelsDir()
        let model = dir.appendingPathComponent(Self.nemotron)
        try require(model)   // bundled pieces.txt is resolved by NemotronStreamingEngine
        // The streaming feed() path (Swift-computed mel) — what `voixful
        // transcribe --streaming` uses. The batch `transcribe(audio:)` needs the
        // baked `preprocess_audio`, which the convert-pipeline palette4 lacks.
        let asr = try await NemotronStreamingEngine(modelURL: model)
        await asr.start()
        _ = try await asr.feed(sampleAudio())
        let transcript = try await asr.finish()
        XCTAssertTrue(transcript.text.lowercased().contains("fox"),
                      "nemotron transcript should contain 'fox': '\(transcript.text)'")
    }

    func testParakeetTranscribes() async throws {
        let dir = try modelsDir()
        let model = dir.appendingPathComponent(Self.parakeet)
        try require(model)
        let engine = try await ParakeetTDTEngine(
            modelURL: model, piecesURL: model.appendingPathComponent("pieces.txt"))
        let text = try await engine.transcribe(audio: sampleAudio())
        XCTAssertTrue(text.lowercased().contains("quick brown fox"),
                      "parakeet transcript: '\(text)'")
    }

    func testCohereTranscribes() async throws {
        let dir = try modelsDir()
        let model = dir.appendingPathComponent(Self.cohere)
        try require(model)
        let engine = try await CohereTranscribeEngine(
            modelURL: model, piecesURL: model.appendingPathComponent("pieces.txt"))
        let text = try await engine.transcribe(audio: sampleAudio())
        XCTAssertTrue(text.lowercased().contains("quick brown fox"),
                      "cohere transcript: '\(text)'")
    }
}
