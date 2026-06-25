// Phase F — 128-dim one-hot language ID construction.
//
// The cache-aware FastConformer encoder takes a 128-dim language one-hot as
// its second input (`lang_id` per `convert/nemotron_convert/convert.py` and
// PLAN.md §2 / §4). The published checkpoint allocates 128 slots for
// language conditioning — that's the prompt-kernel dimensionality, not a
// per-language count. 40 of those slots are populated with the supported
// locales (PLAN.md §2 / Types.swift Language enum); the remaining 88 are
// unused capacity reserved for future locales.
//
// `LangID` is intentionally stateless: there is one canonical mapping from
// `Language` to slot index (the enumeration order of `Language.allCases`),
// and `oneHot(_:)` is a pure function. No actor, no init, no per-instance
// state — just a value type and a static method.

import CoreAI
import Foundation

/// Stateless namespace for building the 128-dim language one-hot vector the
/// encoder's prompt-kernel consumes.
public struct LangID: Sendable {

    /// Prompt-kernel input dimensionality. Hardcoded to 128 per PLAN.md §2;
    /// this is the published checkpoint's value and NOT a knob.
    public static let dim: Int = 128

    /// Build the 128-dim one-hot `NDArray` for `language`, suitable for passing
    /// as the `lang_id` input to `EncoderSession.forward(mel:langId:)`.
    ///
    /// The returned NDArray has shape `[128]` and `.float32` dtype, with
    /// exactly one slot set to 1.0 and the rest zero. The hot slot index is
    /// `Language.allCases.firstIndex(of: language)` — i.e., the declaration
    /// order in `Language` is the canonical mapping from locale → prompt-kernel
    /// slot.
    ///
    /// VERIFY: confirm 128-dim one-hot encoding matches NeMo's prompt-kernel
    /// input expectation (PLAN.md §10 open question). The unknowns are:
    ///   1. Does the model want `[128]` or `[1, 128]` (broadcast over batch)?
    ///      EncoderSession.forward passes whatever this returns directly to
    ///      `inputs: ["lang_id": langId]`; the encoder graph was exported with
    ///      `example_lang` of a specific shape (see
    ///      convert/nemotron_convert/convert.py and the §3 Step 3 sketch).
    ///      If the exported graph wants `[1, 128]`, change the shape literal
    ///      below and the matching `example_lang` in convert.py together.
    ///   2. Is the canonical lang→slot mapping the `Language` declaration order
    ///      (what we use here), or is there a published NeMo lookup table that
    ///      assigns slot indices independently of any Swift-side enum order?
    ///      Phase 1's `.nemo` config inspection (PLAN.md §7 Phase 1) is the
    ///      source of truth; if NeMo's table differs, write a mapping function
    ///      that takes `Language` → `Int` instead of relying on
    ///      `allCases.firstIndex(of:)`, and keep this method's body symmetric
    ///      with `verify_parity.py:_lang_id_one_hot` in convert/.
    ///   3. The 13 broad-coverage and 8 adaptation-ready locales (PLAN.md §2)
    ///      may share slots with their nearest transcription-ready locale
    ///      (e.g., nb-NO might use the same slot as nn-NO) rather than each
    ///      having its own; if so, this function needs a coalescing table.
    ///      Phase 1 task.
    ///
    /// - Parameter language: The locale to encode.
    /// - Returns: A `[128]` `.float32` `NDArray` with the corresponding slot set to 1.0.
    public static func oneHot(_ language: Language) -> NDArray {
        // The slot index for `language` is its position in Language.allCases.
        // Force-unwrap is safe: `language` is an enum case of `Language`, so
        // `firstIndex(of:)` is guaranteed non-nil.
        // VERIFY: see VERIFY (2) above re: NeMo's canonical lang→slot mapping.
        let slot = Language.allCases.firstIndex(of: language)!

        // Allocate the one-hot NDArray. Shape `[dim]` = `[128]` per the
        // model's prompt-kernel input expectation.
        //
        // VERIFY: Core AI API surface — `NDArray(shape:scalarType:)` initializer
        // and `.float32` scalar-type spelling per PLAN.md §4 snippet. Same
        // accessor pattern as EncoderSession/PredictionSession. `var` (not
        // `let`) because we mutate via `mutableView()` below — same convention
        // as EncoderSession's cache buffers at EncoderSession.swift:34-52.
        var array = NDArray(shape: [dim], scalarType: .float32)

        // Write the one-hot pattern: zeros everywhere, 1.0 at `slot`. Uses
        // the `zeroFloat32` / `setFloat32(at:to:)` helpers from
        // NDArrayHelpers.swift (resolves the Phase-D VERIFY on the scalar
        // accessor API). The two-step pattern (zero then set) is intentional:
        // NDArray's initial contents are unspecified, so a fresh allocation
        // must be explicitly zeroed before single-slot writes.
        array.zeroFloat32()
        array.setFloat32(at: slot, to: 1.0)

        return array
    }
}
