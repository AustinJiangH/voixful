// Env-gated compute-unit pinning for AIModel specialization.
//
// `VOIXFUL_COMPUTE=gpu|ane` pins specialization to one compute unit instead of
// the runtime's auto-partitioned default. Motivation: the auto path has shown
// intermittent MPSGraph "invalid parameters / unresolved symbol" failures on
// our exported assets (see convert/nemotron-streaming verify_parity.py, which
// pins GPU for the same reason), and pinning makes placement measurable when
// benchmarking. Default (unset) keeps `.default` — no behavior change.

import CoreAI
import Foundation

/// Specialization options honoring the `VOIXFUL_COMPUTE` env override.
func specializationOptions() -> SpecializationOptions {
    switch ProcessInfo.processInfo.environment["VOIXFUL_COMPUTE"]?.lowercased() {
    case "gpu": return SpecializationOptions(preferredComputeUnitKind: .gpu)
    case "ane", "neuralengine", "neural-engine":
        return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
    case "cpu": return SpecializationOptions(preferredComputeUnitKind: .cpu)
    default: return .default
    }
}

/// Optional separate placement for the decode-side functions (transducer
/// predict/join; Cohere decoder) via `VOIXFUL_COMPUTE_DECODE=cpu|gpu|ane`.
/// nil (unset) = share the primary model instance. The per-call math of the
/// tiny transducer heads is microseconds while an accelerator round-trip is
/// hundreds of µs, so CPU can win there (FluidAudio measured exactly this);
/// a second AIModel instance is the sanctioned per-function placement
/// workaround — Core AI has no per-function options API.
func decodeSpecializationOptions() -> SpecializationOptions? {
    switch ProcessInfo.processInfo.environment["VOIXFUL_COMPUTE_DECODE"]?.lowercased() {
    case "gpu": return SpecializationOptions(preferredComputeUnitKind: .gpu)
    case "ane", "neuralengine", "neural-engine":
        return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
    case "cpu": return SpecializationOptions(preferredComputeUnitKind: .cpu)
    default: return nil
    }
}
