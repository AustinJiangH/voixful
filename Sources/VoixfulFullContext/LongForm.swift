// Long-form segmentation for full-context (batch) engines.
//
// Full-context backends convert to a `.aimodel` whose encoder is traced at a
// FIXED mel window (Parakeet: 1000 frames = 10 s). That is a property of the
// exported asset, NOT of the upstream model — NeMo's Parakeet transcribes ~24
// minutes in one pass. To transcribe audio longer than the exported window
// WITHOUT re-exporting, we do what buffered long-form inference does: cut the
// audio into window-sized chunks at the quietest gaps, transcribe each, and
// concatenate. No audio is dropped and there is no user-facing length cap.
//
// The batch transcriber calls this once at `finalize`; the live wrapper does
// the streaming analog (commit the head as it grows). Both size their chunks
// off `FullContextEngine.maxAudioSamples` — which each engine reports as the
// SMALLER of its encoder window and any downstream decoder token cap (Cohere's
// AED decoder stops at 108 tokens, Canary's at 256), so a chunk can't overrun
// either stage.
//
// Known limitation: cuts land at the quietest 200 ms in the search region. In
// truly gap-free speech (no pause anywhere near the boundary) the cut falls
// mid-word, so the straddling word can be garbled or lost across the join.
// There is no overlap/dedup, so a word is never duplicated. Eliminating this
// entirely needs word-timestamp-aware stitching; for dictation the quietest-gap
// heuristic is a good tradeoff since natural speech pauses far more often than
// once per window (Parakeet ~10 s, Cohere ~15 s).

import Foundation
import VoixfulEngine

public enum LongForm {

    /// Fraction of the engine window a chunk is allowed to fill. Headroom
    /// absorbs mel-frame rounding (STFT centering can add a frame or two) so a
    /// chunk never spills past the window and truncates.
    public static let fillFraction = 0.9

    /// Cut `audio` into ascending, gap-free ranges each no larger than
    /// `chunk` samples, preferring the quietest 200 ms near each boundary so
    /// cuts land between words rather than through them. Returns a single
    /// full-length range when the audio already fits in one chunk.
    public static func segments(_ audio: [Float], chunk: Int) -> [Range<Int>] {
        guard chunk > 0, audio.count > chunk else {
            return audio.isEmpty ? [] : [0..<audio.count]
        }
        var ranges: [Range<Int>] = []
        var start = 0
        while audio.count - start > chunk {
            let hardEnd = start + chunk
            // Search the last quarter of the chunk for a pause; never past the
            // hard window limit, and never before the halfway mark so every
            // chunk makes real progress even in unbroken speech.
            let cut = quietestSplit(audio, from: start + chunk * 3 / 4, to: hardEnd)
            let safeCut = min(max(cut, start + chunk / 2), hardEnd)
            ranges.append(start..<safeCut)
            start = safeCut
        }
        ranges.append(start..<audio.count)
        return ranges
    }

    /// Transcribe arbitrarily long `audio` through a fixed-window engine by
    /// segmenting on silence and joining the per-chunk text. One pass when the
    /// audio fits the window.
    public static func transcribe<Engine: FullContextEngine>(
        audio: [Float], with engine: Engine
    ) async throws -> String {
        let chunk = Int(Double(engine.maxAudioSamples) * fillFraction)
        let ranges = segments(audio, chunk: chunk)
        var parts: [String] = []
        for range in ranges {
            let text = try await engine.transcribe(audio: Array(audio[range])).text
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty { parts.append(text) }
        }
        return parts.joined(separator: " ")
    }

    /// Index of the quietest 200 ms in `[from, to)` — a cheap pause detector so
    /// segment cuts fall between words. Returns the middle of the quietest
    /// window, or the clamped end when the span is too small to scan.
    public static func quietestSplit(_ audio: [Float], from: Int, to: Int) -> Int {
        let win = 3_200  // 200 ms at 16 kHz
        let lo = max(0, from)
        let hi = min(to, audio.count) - win
        guard hi > lo else { return min(to, audio.count) }
        var best = lo
        var bestEnergy = Float.greatestFiniteMagnitude
        var i = lo
        while i < hi {
            var e: Float = 0
            for j in i..<(i + win) { e += audio[j] * audio[j] }
            if e < bestEnergy {
                bestEnergy = e
                best = i
            }
            i += win / 2
        }
        return best + win / 2
    }
}
