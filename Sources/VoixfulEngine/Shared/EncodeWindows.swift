// Discovery of multi-window encoder functions in an `.aimodel`.
//
// The convert pipelines export the encoder at several fixed mel-frame windows —
// the traced graph always runs its full window, so a short clip inside a long
// window burns compute on zero padding. The canonical `encode` is the LARGEST
// window; smaller variants are named `encode_<frames>`. Engines pick the
// smallest window that fits the clip. Single-window assets simply have no
// `encode_<frames>` functions and keep their old behavior.

import CoreAI
import Foundation

/// Load every `encode_<frames>` variant from `model`, ascending by window size.
/// Returns `[]` for single-window assets.
func loadEncodeWindowFunctions(
    model: AIModel, modelURL: URL
) throws -> [(frames: Int, fn: InferenceFunction)] {
    let names = try ModelBackend.assetFunctions(modelURL: modelURL).names
    var windows: [(frames: Int, fn: InferenceFunction)] = []
    for name in names {
        guard name.hasPrefix("encode_"),
              let frames = Int(name.dropFirst("encode_".count)) else { continue }
        guard let fn = try model.loadFunction(named: name) else {
            throw ASRError.missingFunction(name)
        }
        windows.append((frames, fn))
    }
    return windows.sorted { $0.frames < $1.frames }
}

/// The smallest window that fits `realFrames`, else `nil` (use the canonical
/// full-window `encode`).
func pickEncodeWindow(
    _ windows: [(frames: Int, fn: InferenceFunction)], realFrames: Int
) -> (frames: Int, fn: InferenceFunction)? {
    windows.first { $0.frames >= realFrames }
}
