// Unit tests for ModelBackend.classify — pure function over function-name
// sets, no assets needed. Locks in the detection ordering so adding a new
// backend can't silently re-route an existing asset (the Granite rule matches
// FIRST because its encode_<F> functions are stateless like Parakeet's).

import XCTest
@testable import VoixfulEngine

final class BackendClassifyTests: XCTestCase {
    private func fns(_ names: Set<String>, encodeHasStates: Bool = false) -> ModelBackend.AssetFunctions {
        ModelBackend.AssetFunctions(names: names, encodeHasStates: encodeHasStates)
    }

    func testQwen3ByStepEntrypoint() {
        // Qwen3-ASR ships a distinct `step_qwen3` AR entrypoint; its stateless
        // encode_<T> functions must not shadow it down to Parakeet.
        XCTAssertEqual(
            ModelBackend.classify(fns(
                ["encode_1000", "embed_256", "prefill_asr_160", "step_qwen3"])),
            .qwen3)
    }

    func testCanaryByStepEntrypoint() {
        XCTAssertEqual(
            ModelBackend.classify(fns(
                ["encode_1000", "embed_256", "prefill_asr_160", "step_asr", "prefill_llm_256", "step_llm"])),
            .canary)
    }

    func testGraniteByEditEntrypoint() {
        XCTAssertEqual(
            ModelBackend.classify(fns(["encode_200", "encode_400", "edit_40_64", "edit_80_96"])),
            .granite)
    }

    func testCohereByAnyDecodeEntrypoint() {
        XCTAssertEqual(ModelBackend.classify(fns(["encode", "decode_x4", "project_cross_kv"])), .cohere)
        XCTAssertEqual(ModelBackend.classify(fns(["encode", "decode_step"])), .cohere)
    }

    func testNemotronByEncodeStates() {
        XCTAssertEqual(
            ModelBackend.classify(fns(["encode", "predict_step", "join"], encodeHasStates: true)),
            .nemotron)
    }

    func testParakeetFallback() {
        XCTAssertEqual(
            ModelBackend.classify(fns(["encode", "encode_125", "predict_step", "join", "project_enc"])),
            .parakeet)
    }
}
