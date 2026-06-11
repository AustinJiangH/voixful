// Shared VAD (Voice Activity Detection) defaults — single source of truth for
// the energy-based VAD used by the CLI `mic` subcommand and the SwiftUI demo's
// TranscriptionViewModel. Resolves the Phase-L finding that these values were
// duplicated across two sites (Sources/NemotronASRCLI/main.swift and
// Demos/NemotronDemo/Sources/NemotronDemo/TranscriptionViewModel.swift).
//
// All values are placeholders per PLAN.md §7 Phase 7 / §10 open question on VAD
// strategy. Tune once a real mic recording surfaces the miss rate, OR replace
// the whole energy-based path with a Silero Core AI conversion (PLAN.md §10
// follow-up).

import Foundation

/// Defaults for the energy-based VAD used in live capture paths.
public struct VADDefaults: Sendable {
    /// Samples per VAD analysis window. 320 = 20 ms @ 16 kHz.
    public static let windowSamples: Int = 320

    /// RMS threshold below which a window counts as silence. Placeholder —
    /// tune once real mic recordings surface the miss rate.
    public static let energyThreshold: Float = 0.005

    /// Wall-clock duration of one VAD analysis window, in milliseconds.
    public static let windowMS: Double = 20.0

    /// Total post-speech silence required to treat the utterance as complete.
    public static let utteranceEndSilenceMS: Double = 500.0

    /// Number of consecutive silent windows that trigger utterance end.
    /// Derived from `utteranceEndSilenceMS / windowMS`.
    public static var silenceWindowsToFinish: Int {
        Int(utteranceEndSilenceMS / windowMS)
    }
}
