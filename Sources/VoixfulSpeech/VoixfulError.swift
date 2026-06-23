// VoixfulSpeech — brand-level errors.
//
// Model-agnostic: this target never imports Core AI or any model runtime.

import Foundation

/// Errors surfaced by the Voixful analyzer + module layer.
public enum VoixfulError: Error, Sendable {
    /// The requested `Locale` is not supported by the chosen model backend.
    /// Associated value is the BCP-47 tag we tried to resolve.
    case unsupportedLocale(String)

    /// Audio could not be converted to the backend's required PCM format.
    case audioConversionFailed(String)

    /// The underlying model engine failed (load, inference, finalize).
    case engine(String)

    /// A module method was called before `VoixfulAnalyzer.start(inputSequence:)`.
    case notStarted
}
