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
    default: return .default
    }
}
