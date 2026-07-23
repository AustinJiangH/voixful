// Mel parity gate.
// Loads a fixture WAV + a NeMo-generated reference [128 × T] float32 blob and asserts
// max-abs / mean-abs deltas. Fixtures live under Tests/VoixfulEngineTests/fixtures/;
// see fixtures/README.md for the generation recipe.

import XCTest
import Foundation
import AVFoundation
@testable import VoixfulEngine

final class MelParityTests: XCTestCase {

    func testMelMatchesNemoReference() async throws {
        // Bundle.module requires a `resources:` block on the test target in
        // Package.swift; with no fixtures committed yet that block is
        // unnecessary, so we always resolve via the #filePath fallback. If
        // fixtures are added later AND Package.swift is amended to copy them in,
        // either path stays correct.
        let wavURL = Self.fallbackFixturesDir().appendingPathComponent("sample.wav")
        let refURL = Self.fallbackFixturesDir().appendingPathComponent("mel_reference.bin")

        guard FileManager.default.fileExists(atPath: wavURL.path) else {
            throw XCTSkip("fixture missing — generate per fixtures/README.md (need sample.wav)")
        }
        guard FileManager.default.fileExists(atPath: refURL.path) else {
            throw XCTSkip("fixture missing — generate per fixtures/README.md (need mel_reference.bin)")
        }

        let samples = try Self.loadWAVMono16k(at: wavURL)
        let reference = try Self.loadFloat32Blob(at: refURL)

        let extractor = MelExtractor()
        let mel = await extractor.extract(samples: samples)

        XCTAssertEqual(mel.count, reference.count,
                       "mel length mismatch (got \(mel.count), reference \(reference.count))")

        var maxAbs: Float = 0
        var sumAbs: Double = 0
        for i in 0..<mel.count {
            let d = abs(mel[i] - reference[i])
            if d > maxAbs { maxAbs = d }
            sumAbs += Double(d)
        }
        let meanAbs = Float(sumAbs / Double(mel.count))

        XCTAssertLessThan(maxAbs, 1e-3, "max-abs delta \(maxAbs) exceeds 1e-3")
        XCTAssertLessThan(meanAbs, 1e-5, "mean-abs delta \(meanAbs) exceeds 1e-5")
    }

    // MARK: - Helpers

    /// Bundle.module isn't always populated when fixtures aren't declared as resources
    /// in Package.swift; this fallback resolves to Tests/VoixfulEngineTests/fixtures/
    /// relative to the test source file. VERIFY: once resources are wired into
    /// Package.swift via `resources: [.copy("fixtures")]`, this fallback becomes dead code.
    private static func fallbackFixturesDir() -> URL {
        // #filePath points at this file in DEBUG; resolve siblings.
        let here = URL(fileURLWithPath: #filePath)
        return here.deletingLastPathComponent().appendingPathComponent("fixtures")
    }

    /// Load a WAV file at 16 kHz mono Float32. Resamples via AVAudioConverter if needed.
    private static func loadWAVMono16k(at url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let target = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                   sampleRate: 16_000,
                                   channels: 1,
                                   interleaved: false)!
        let srcFormat = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)

        guard let srcBuf = AVAudioPCMBuffer(pcmFormat: srcFormat, frameCapacity: frameCount) else {
            throw NSError(domain: "MelParityTests", code: 1)
        }
        try file.read(into: srcBuf)

        let converter = AVAudioConverter(from: srcFormat, to: target)!
        let ratio = target.sampleRate / srcFormat.sampleRate
        let outFrames = AVAudioFrameCount(Double(srcBuf.frameLength) * ratio) + 1024
        guard let dstBuf = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: outFrames) else {
            throw NSError(domain: "MelParityTests", code: 2)
        }

        var didFeed = false
        let status = converter.convert(to: dstBuf, error: nil) { _, outStatus in
            if didFeed {
                outStatus.pointee = .endOfStream
                return nil
            }
            didFeed = true
            outStatus.pointee = .haveData
            return srcBuf
        }
        guard status != .error else {
            throw NSError(domain: "MelParityTests", code: 3)
        }

        let n = Int(dstBuf.frameLength)
        guard let channel = dstBuf.floatChannelData?[0] else {
            throw NSError(domain: "MelParityTests", code: 4)
        }
        return Array(UnsafeBufferPointer(start: channel, count: n))
    }

    /// Read a little-endian float32 blob into `[Float]`. The reference is row-major
    /// `[nMels × T]`; this loader is shape-agnostic and the caller asserts cardinality.
    private static func loadFloat32Blob(at url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        let count = data.count / MemoryLayout<Float>.size
        var out = [Float](repeating: 0, count: count)
        _ = out.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst, count: count * MemoryLayout<Float>.size)
        }
        return out
    }
}
