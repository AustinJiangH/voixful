// Helpers for the Core AI `NDArray` accessor surface.
//
// Core AI's NDArray exposes its element memory through `~Copyable`/`~Escapable`
// views (`MutableView<Element>`, `View<Element>`). The closure-based accessors
// take a tuple `(pointer, shape, strides)` rather than a flat buffer, so a few
// common operations — fill-with-zero, single-element write, argmax — get
// repetitive at the call site. These extensions centralize them.
//
// The Core AI `NDArray.shape` is a `[Int]` (host-side property on the array
// itself), distinct from the `MutableView.shape` which is a `Span<Int>`. We use
// the host-side shape for total-element-count math because it avoids reading
// the borrowed span inside the closure body.
//
// Verified against the macOS 27.0 SDK Core AI swiftinterface (2026-06-11).

import CoreAI
import Foundation

@available(macOS 27.0, *)
extension NDArray {
    /// Total number of scalars in the array (the product of every shape dim).
    @inlinable
    var elementCount: Int {
        var n = 1
        for dim in self.shape { n *= dim }
        return n
    }

    /// Zero every Float32 element, in place. Used at utterance boundaries to
    /// reset stateful caches without reallocating the backing buffer (which
    /// would invalidate the existing `MutableViews` registration).
    mutating func zeroFloat32() {
        let count = self.elementCount
        var mv = self.mutableView(as: Float.self)
        mv.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: 0, count: count)
        }
    }

    /// Zero every Float16 element, in place. Used by the Canary-Qwen fp16 KV
    /// caches (fp16 state I/O lets the AR decoder land on the ANE).
    mutating func zeroFloat16() {
        let count = self.elementCount
        var mv = self.mutableView(as: Float16.self)
        mv.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: 0, count: count)
        }
    }

    /// Zero every Int32 element, in place. Mirror of `zeroFloat32` for the
    /// `cache_last_channel_len` int32 counter cache.
    mutating func zeroInt32() {
        let count = self.elementCount
        var mv = self.mutableView(as: Int32.self)
        mv.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: 0, count: count)
        }
    }

    /// Zero every Int64 element, in place. The real model's
    /// `cache_last_channel_len` is int64 (verified against NeMo's encoder).
    mutating func zeroInt64() {
        let count = self.elementCount
        var mv = self.mutableView(as: Int64.self)
        mv.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: 0, count: count)
        }
    }

    /// Write a single Float32 scalar at flat index `index`. Caller is
    /// responsible for ensuring the index is in bounds; this is a hot-path
    /// helper used by `LangID.oneHot` to set a single 1.0 slot.
    mutating func setFloat32(at index: Int, to value: Float) {
        var mv = self.mutableView(as: Float.self)
        mv.withUnsafeMutablePointer { ptr, _, _ in
            ptr[index] = value
        }
    }

    /// Write a single Int32 scalar at flat index `index`. Used by
    /// `PredictionSession.step` to load the previous token into the
    /// pre-allocated `prev_token` scratch NDArray.
    mutating func setInt32(at index: Int, to value: Int32) {
        var mv = self.mutableView(as: Int32.self)
        mv.withUnsafeMutablePointer { ptr, _, _ in
            ptr[index] = value
        }
    }

    /// Fill every Float32 element with `value`, in place. Used by tests to
    /// write a non-zero sentinel pattern before verifying that `reset()` zeros
    /// it back.
    mutating func fillFloat32(_ value: Float) {
        let count = self.elementCount
        var mv = self.mutableView(as: Float.self)
        mv.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: value, count: count)
        }
    }

    /// Fill every Int64 element with `value`, in place.
    mutating func fillInt64(_ value: Int64) {
        let count = self.elementCount
        var mv = self.mutableView(as: Int64.self)
        mv.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: value, count: count)
        }
    }

    /// Snapshot an Int64 NDArray as a flat `[Int64]`.
    func flatInt64() -> [Int64] {
        let count = self.elementCount
        var out = [Int64](repeating: 0, count: count)
        self.view(as: Int64.self).withUnsafePointer { ptr, _, _ in
            for i in 0..<count { out[i] = ptr[i] }
        }
        return out
    }

    /// Fill every Int32 element with `value`, in place. Used by tests for the
    /// int32 cache (`cache_last_channel_len`) sentinel writes.
    mutating func fillInt32(_ value: Int32) {
        let count = self.elementCount
        var mv = self.mutableView(as: Int32.self)
        mv.withUnsafeMutablePointer { ptr, _, _ in
            ptr.update(repeating: value, count: count)
        }
    }

    /// Snapshot a Float32 NDArray as a flat `[Float]`. Used by tests to take a
    /// read-only copy of buffer contents for assertions.
    func flatFloat32() -> [Float] {
        let count = self.elementCount
        var out = [Float](repeating: 0, count: count)
        self.view(as: Float.self).withUnsafePointer { ptr, _, _ in
            for i in 0..<count { out[i] = ptr[i] }
        }
        return out
    }

    /// Snapshot an Int32 NDArray as a flat `[Int32]`.
    func flatInt32() -> [Int32] {
        let count = self.elementCount
        var out = [Int32](repeating: 0, count: count)
        self.view(as: Int32.self).withUnsafePointer { ptr, _, _ in
            for i in 0..<count { out[i] = ptr[i] }
        }
        return out
    }

    /// Argmax over a Float32 NDArray viewed as a flat buffer. Used by
    /// `JoinerSession.argmax` to find the highest-logit token id per step of
    /// the RNN-T greedy decode.
    ///
    /// Returns the flat index of the maximum, or 0 if the array is empty.
    /// Caller is responsible for shape semantics (typically the logits are
    /// `[1, 1, 1, vocab]` so flat index == vocab id).
    func argmaxFloat32() -> Int32 {
        let count = self.elementCount
        guard count > 0 else { return 0 }
        var bestIndex: Int32 = 0
        var bestValue: Float = -.infinity
        let v = self.view(as: Float.self)
        v.withUnsafePointer { ptr, _, _ in
            for i in 0..<count {
                let val = ptr[i]
                if val > bestValue {
                    bestValue = val
                    bestIndex = Int32(i)
                }
            }
        }
        return bestIndex
    }
}
