// Real-asset check that the 30 s Parakeet window works end to end: the engine
// reads the window from parakeet.json, and a 22 s clip transcribes in ONE pass
// (no segmentation) without truncation — the property the 1000-frame window
// couldn't provide.
//
// Gated on VOIXFUL_TEST_MODELS_DIR containing the 30 s asset; skipped otherwise
// (like LiveLongFormTests / BackendIntegrationTests).

import AVFoundation
import XCTest
import VoixfulEngine
@testable import VoixfulSpeech

final class ParakeetWindowIntegrationTests: XCTestCase {

    private func modelURL() throws -> URL {
        guard let dir = ProcessInfo.processInfo.environment["VOIXFUL_TEST_MODELS_DIR"], !dir.isEmpty else {
            throw XCTSkip("set VOIXFUL_TEST_MODELS_DIR to run")
        }
        let url = URL(fileURLWithPath: dir)
            .appendingPathComponent("parakeet-tdt-0.6b-v3-palette4-30s.aimodel")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("30 s parakeet asset not present at \(url.path)")
        }
        return url
    }

    private func fixtureSamples(repeats: Int) throws -> [Float] {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("fixtures/sample.wav")
        let file = try AVAudioFile(forReading: fixture)
        let buf = AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                   frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: buf)
        let one = try AudioInputConverter().convert(buf)
        var out: [Float] = []
        for _ in 0..<repeats { out.append(contentsOf: one) }
        return out
    }

    func testManifestGivesThirtySecondWindow() async throws {
        let url = try modelURL()
        let engine = try await ParakeetTDTEngine(
            modelURL: url, piecesURL: url.appendingPathComponent("pieces.txt"))
        // 3000 mel frames × 160-sample hop = 30 s.
        XCTAssertEqual(engine.maxAudioSamples, 3000 * 160)
    }

    func testTwentyTwoSecondClipTranscribesInOnePassWithoutTruncation() async throws {
        let url = try modelURL()
        let engine = try await ParakeetTDTEngine(
            modelURL: url, piecesURL: url.appendingPathComponent("pieces.txt"))

        // ~22 s (sample.wav ≈ 5.5 s × 4) — well past the old 1000-frame (10 s)
        // window, but within the 30 s window, so ONE engine pass must return the
        // whole thing. sample.wav says "...the quick brown fox..." once per rep.
        let audio = try fixtureSamples(repeats: 4)
        XCTAssertGreaterThan(audio.count, 20 * 16_000, "clip should exceed the old window")
        XCTAssertLessThanOrEqual(audio.count, engine.maxAudioSamples, "clip must fit one 30 s pass")

        let text = try await engine.transcribe(audio: audio).text.lowercased()
        let foxes = text.components(separatedBy: "fox").count - 1
        XCTAssertEqual(foxes, 4,
            "30 s single pass dropped content: \(foxes)/4 reps in: \(text)")
    }
}
