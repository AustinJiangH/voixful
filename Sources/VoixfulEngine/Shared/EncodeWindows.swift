// Discovery of multi-window encoder functions in an `.aimodel`.
//
// The convert pipelines export the encoder at several fixed mel-frame windows —
// the traced graph always runs its full window, so a short clip inside a long
// window burns compute on zero padding. The canonical `encode` is the LARGEST
// window; smaller variants are named `encode_<frames>`. Engines pick the
// smallest window that fits the clip. Single-window assets simply have no
// `encode_<frames>` functions and keep their old behavior.
//
// IMPORTANT: these return NAMES, not loaded functions. `AIModel.loadFunction`
// specializes the function's graph — for a multi-GB fp16 model, eagerly loading
// every window materializes one specialized weight set PER WINDOW (measured
// ~52 GB peak RSS for the 2B fp16 Cohere with 4 windows). Engines load lazily,
// only the function a clip actually uses.

import Foundation

/// Parse every `encode_<frames>` variant from an asset's function names,
/// ascending by window size. Returns `[]` for single-window assets.
func encodeWindowNames(_ names: Set<String>) -> [(frames: Int, name: String)] {
    var windows: [(frames: Int, name: String)] = []
    for name in names {
        guard name.hasPrefix("encode_"),
              let frames = Int(name.dropFirst("encode_".count)) else { continue }
        windows.append((frames, name))
    }
    return windows.sorted { $0.frames < $1.frames }
}

/// The smallest window that fits `realFrames`, else `nil` (use the canonical
/// full-window `encode`).
func pickEncodeWindow(
    _ windows: [(frames: Int, name: String)], realFrames: Int
) -> (frames: Int, name: String)? {
    windows.first { $0.frames >= realFrames }
}
