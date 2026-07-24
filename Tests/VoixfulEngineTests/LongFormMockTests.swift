// Deterministic long-form segmentation tests — NO model required, so these
// run in CI unconditionally (unlike the fixture-gated LiveLongFormTests).
//
// A mock FullContextEngine with a small `maxAudioSamples` window records every
// audio length it is handed and decodes synthetic "words" (runs of a distinct
// non-zero value, separated by silence). Feeding audio far longer than the
// window through both the batch and live transcribers must:
//   1. transcribe EVERY word (no truncation past the window), and
//   2. never hand any single pass more than one window of audio.
// Together those are exactly the property the 10 s-truncation bug violated.

import AVFoundation
import XCTest
import VoixfulEngine
import VoixfulFullContext
@testable import VoixfulSpeech

// Thread-safe recorder shared by all instances of the mock (the transcribers
// construct their own engine internally, so the test can't reach the instance).
private final class MockRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var _maxSeen = 0
    private var _passes = 0
    func record(_ n: Int) { lock.lock(); _maxSeen = max(_maxSeen, n); _passes += 1; lock.unlock() }
    var maxSeen: Int { lock.lock(); defer { lock.unlock() }; return _maxSeen }
    var passes: Int { lock.lock(); defer { lock.unlock() }; return _passes }
    func reset() { lock.lock(); _maxSeen = 0; _passes = 0; lock.unlock() }
}
private let mockRecorder = MockRecorder()

private actor MockFullContext: FullContextEngine {
    static let displayName = "Mock"
    static let identifierSuffix = "mock"
    static var supportedLocales: [Locale] { [Locale(identifier: "en-US")] }

    // 1 s window at 16 kHz — small so a few seconds of audio must segment.
    nonisolated let maxAudioSamples = 16_000

    init(modelURL: URL, piecesURL: URL) async throws {}
    func prewarm() {}

    // Decode: each maximal run of a distinct non-zero value is one word "w<v>";
    // silence (0) separates words. Collapses the run so a word yields one token.
    func transcribe(audio: [Float]) async throws -> Transcript {
        mockRecorder.record(audio.count)
        var tokens: [String] = []
        var prev: Float = 0
        for v in audio {
            if v != 0, v != prev { tokens.append("w\(Int(v))") }
            prev = v
        }
        return Transcript(text: tokens.joined(separator: " "), segments: [], language: nil)
    }
}

// A mock whose window is small enough (0.5 s) that the FIRST refresh is a
// head-commit, and which throws on chosen transcribe calls to exercise the
// error path: a failed commit must retry, never drop the head audio.
private struct MockTransientError: Error {}
private final class ThrowBox: @unchecked Sendable {
    private let lock = NSLock()
    private var _calls = 0
    private var _throwOn: Set<Int> = []
    func reset(throwOn: Set<Int>) { lock.lock(); _calls = 0; _throwOn = throwOn; lock.unlock() }
    func shouldThrow() -> Bool { lock.lock(); defer { lock.unlock() }; _calls += 1; return _throwOn.contains(_calls) }
}
private let throwBox = ThrowBox()

private actor ThrowingFullContext: FullContextEngine {
    static let displayName = "ThrowMock"
    static let identifierSuffix = "throwmock"
    static var supportedLocales: [Locale] { [Locale(identifier: "en-US")] }
    nonisolated let maxAudioSamples = 8_000   // 0.5 s → refreshStride (0.7 s) forces commit first

    init(modelURL: URL, piecesURL: URL) async throws {}
    func prewarm() {}
    func transcribe(audio: [Float]) async throws -> Transcript {
        if throwBox.shouldThrow() { throw MockTransientError() }
        var tokens: [String] = []
        var prev: Float = 0
        for v in audio {
            if v != 0, v != prev { tokens.append("w\(Int(v))") }
            prev = v
        }
        return Transcript(text: tokens.joined(separator: " "), segments: [], language: nil)
    }
}

final class LongFormMockTests: XCTestCase {

    private let wordWidth = 1_600   // 0.1 s
    private let gapWidth = 3_200    // 0.2 s silence — wider than a word so cuts land here

    /// `wordCount` words of distinct value 1…N, each separated by silence.
    private func makeWords(_ wordCount: Int) -> [Float] {
        var audio: [Float] = []
        for k in 1...wordCount {
            audio.append(contentsOf: [Float](repeating: Float(k), count: wordWidth))
            audio.append(contentsOf: [Float](repeating: 0, count: gapWidth))
        }
        return audio
    }

    // MARK: pure segmentation

    func testSegmentsAreContiguousAndBounded() {
        let audio = makeWords(30)              // ≈ 9 s, window is 1 s
        let chunk = Int(Double(16_000) * LongForm.fillFraction)
        let ranges = LongForm.segments(audio, chunk: chunk)
        XCTAssertGreaterThan(ranges.count, 1, "long audio should split into multiple chunks")
        XCTAssertEqual(ranges.first?.lowerBound, 0)
        XCTAssertEqual(ranges.last?.upperBound, audio.count, "segments must cover to the end")
        var cursor = 0
        for r in ranges {
            XCTAssertEqual(r.lowerBound, cursor, "segments must be gap-free and non-overlapping")
            XCTAssertLessThanOrEqual(r.count, chunk, "no segment may exceed the window")
            cursor = r.upperBound
        }
    }

    func testShortAudioIsOneSegment() {
        let audio = [Float](repeating: 1, count: 8_000)
        XCTAssertEqual(LongForm.segments(audio, chunk: 14_400), [0..<8_000])
        XCTAssertEqual(LongForm.segments([], chunk: 14_400), [])
    }

    // MARK: batch path

    func testBatchTranscribesEveryWordPastTheWindow() async throws {
        mockRecorder.reset()
        let audio = makeWords(30)              // ≈ 9 s through a 1 s window
        let transcriber = FullContextTranscriber<MockFullContext>(
            modelURL: URL(fileURLWithPath: "/dev/null"),
            piecesURL: URL(fileURLWithPath: "/dev/null"))
        try await transcriber.prepare()

        let collector = Task { () throws -> String? in
            var final: String?
            for try await r in transcriber.results where r.isFinal { final = String(r.text.characters) }
            return final
        }
        try await feed(audio, into: transcriber)
        try await transcriber.finalize()
        let text = try await collector.value ?? ""

        assertAllWordsPresent(text, count: 30)
        XCTAssertLessThanOrEqual(mockRecorder.maxSeen, 16_000,
            "a single pass exceeded the engine window — would truncate on a real engine")
    }

    // MARK: live path

    func testLiveTranscribesEveryWordPastTheWindow() async throws {
        mockRecorder.reset()
        let audio = makeWords(30)
        let live = LiveFullContextTranscriber<MockFullContext>(
            modelURL: URL(fileURLWithPath: "/dev/null"),
            piecesURL: URL(fileURLWithPath: "/dev/null"))
        try await live.prepare()

        let collector = Task { () throws -> String? in
            var final: String?
            for try await r in live.results where r.isFinal { final = String(r.text.characters) }
            return final
        }
        try await feed(audio, into: live)
        try await live.finalize()
        let text = try await collector.value ?? ""

        assertAllWordsPresent(text, count: 30)
        XCTAssertLessThanOrEqual(mockRecorder.maxSeen, 16_000,
            "a single live pass exceeded the engine window — would truncate on a real engine")
    }

    func testLiveRetriesInsteadOfDroppingWhenCommitTranscribeThrows() async throws {
        throwBox.reset(throwOn: [1])   // fail the very first (commit) transcribe
        let audio = makeWords(20)
        let live = LiveFullContextTranscriber<ThrowingFullContext>(
            modelURL: URL(fileURLWithPath: "/dev/null"),
            piecesURL: URL(fileURLWithPath: "/dev/null"))
        try await live.prepare()

        let collector = Task { () throws -> String? in
            var final: String?
            for try await r in live.results where r.isFinal { final = String(r.text.characters) }
            return final
        }
        try await feed(audio, into: live)
        try await live.finalize()
        let text = try await collector.value ?? ""

        // Before the fix, the thrown commit still dropped its head samples →
        // the leading words vanished. The retry must recover every word.
        assertAllWordsPresent(text, count: 20)
    }

    // MARK: helpers

    private func feed(_ audio: [Float], into t: any Transcribing) async throws {
        let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000,
                                channels: 1, interleaved: false)!
        var i = 0
        while i < audio.count {
            let n = min(8_000, audio.count - i)   // 0.5 s buffers, like a mic tap
            let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(n))!
            buf.frameLength = AVAudioFrameCount(n)
            audio[i..<(i + n)].withUnsafeBufferPointer { src in
                buf.floatChannelData![0].update(from: src.baseAddress!, count: n)
            }
            try await t.consume(AnalyzerInput(buffer: buf))
            i += n
        }
    }

    private func assertAllWordsPresent(_ text: String, count: Int) {
        let tokens = Set(text.split(separator: " ").map(String.init))
        let missing = (1...count).map { "w\($0)" }.filter { !tokens.contains($0) }
        XCTAssertTrue(missing.isEmpty, "dropped words past the window: \(missing) — got: \(text)")
    }
}
