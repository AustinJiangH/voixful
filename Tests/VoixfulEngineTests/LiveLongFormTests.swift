// Long-dictation behavior of LiveFullContextTranscriber: audio beyond the
// engine's largest window must NOT be dropped. The wrapper commits the head
// at a silence gap once the live buffer passes its threshold and keeps
// re-transcribing only the tail; the final is committed text + final tail.
//
// Gated on VOIXFUL_TEST_MODELS_DIR (needs the real Granite asset) like
// BackendIntegrationTests.

import AVFoundation
import XCTest
import VoixfulEngine
import VoixfulFullContext
import VoixfulGraniteNAR
@testable import VoixfulSpeech

final class LiveLongFormTests: XCTestCase {
    func testGraniteLiveCommitsBeyondLargestWindow() async throws {
        guard let dir = ProcessInfo.processInfo.environment["VOIXFUL_TEST_MODELS_DIR"], !dir.isEmpty else {
            throw XCTSkip("set VOIXFUL_TEST_MODELS_DIR to run")
        }
        let modelURL = URL(fileURLWithPath: dir)
            .appendingPathComponent("granite-speech-4.1-2b-nar-palette4.aimodel")
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw XCTSkip("granite asset not present")
        }
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("fixtures/sample.wav")
        let file = try AVAudioFile(forReading: fixture)
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
                                channels: 1, interleaved: false)!
        let clipBuf = AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                       frameCapacity: AVAudioFrameCount(file.length))!
        try file.read(into: clipBuf)

        // 7 repetitions + 0.4 s silence gaps ≈ 41 s — past the 32 s window.
        let converter = AudioInputConverter()
        let clip = try converter.convert(clipBuf)
        var long: [Float] = []
        for _ in 0..<7 {
            long.append(contentsOf: clip)
            long.append(contentsOf: [Float](repeating: 0, count: 6_400))
        }

        let live = LiveFullContextTranscriber<GraniteNAREngine>(
            modelURL: modelURL, piecesURL: modelURL.appendingPathComponent("pieces.txt"))
        try await live.prepare()

        let collector = Task { () throws -> String? in
            var final: String?
            for try await r in live.results where r.isFinal {
                final = String(r.text.characters)
            }
            return final
        }
        // Feed in 0.5 s chunks like a mic tap would.
        var i = 0
        while i < long.count {
            let n = min(8_000, long.count - i)
            let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(n))!
            buf.frameLength = AVAudioFrameCount(n)
            long[i..<(i + n)].withUnsafeBufferPointer { src in
                buf.floatChannelData![0].update(from: src.baseAddress!, count: n)
            }
            try await live.consume(AnalyzerInput(buffer: buf))
            i += n
        }
        try await live.finalize()
        let text = try await collector.value ?? ""

        // Every repetition must survive: count "fox" occurrences.
        let foxes = text.components(separatedBy: "fox").count - 1
        XCTAssertGreaterThanOrEqual(
            foxes, 6,
            "long dictation lost content past the window: only \(foxes)/7 repetitions in: \(text)")
    }
}
