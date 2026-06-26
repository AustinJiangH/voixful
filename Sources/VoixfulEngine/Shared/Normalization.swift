// Shared feature normalization for the NeMo `normalize: per_feature` preprocessor.
//
// Both the Parakeet and Cohere encoders consume per-feature-normalized log-mel
// (the Nemotron streaming path uses `normalize: NA` and skips this). Kept in one
// place so the two backends can't drift.

import Foundation

/// NeMo `normalize_batch(..., "per_feature")`: standardize each of `nMels` mel
/// bins to zero mean / unit standard deviation over `frames` (unbiased std, with
/// a `+1e-5` guard). Input and output are row-major `[nMels × frames]`.
///
/// Returns the input unchanged when `frames < 2` (variance is undefined).
func perFeatureNormalize(_ mel: [Float], nMels: Int, frames: Int) -> [Float] {
    guard frames > 1 else { return mel }
    var out = mel
    let n = Float(frames)
    for m in 0..<nMels {
        let base = m * frames
        var mean: Float = 0
        for t in 0..<frames { mean += mel[base + t] }
        mean /= n
        var m2: Float = 0
        for t in 0..<frames { let d = mel[base + t] - mean; m2 += d * d }
        let std = (m2 / (n - 1)).squareRoot() + 1e-5
        for t in 0..<frames { out[base + t] = (mel[base + t] - mean) / std }
    }
    return out
}
