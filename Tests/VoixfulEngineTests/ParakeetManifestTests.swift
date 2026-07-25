// parakeet.json parsing — the manifest that carries the encoder's single-pass
// window. Deterministic, no model: exercises the fallback contract that keeps
// pre-manifest (10 s) assets working and rejects malformed manifests wholesale
// so the decode paths never see a window/projector mismatch.

import XCTest
@testable import VoixfulEngine

final class ParakeetManifestTests: XCTestCase {

    private func parse(_ json: String) -> (melFrames: Int, projT: Int) {
        ParakeetTDTEngine.parseWindowConfig(Data(json.utf8))
    }

    func testValid30sManifest() {
        let r = parse(#"{"format":1,"mel_frames":3000,"proj_t":375,"windows":[1000,3000]}"#)
        XCTAssertEqual(r.melFrames, 3000)
        XCTAssertEqual(r.projT, 375)
    }

    func testValid10sManifestMatchesTheDefault() {
        let r = parse(#"{"mel_frames":1000,"proj_t":125}"#)
        XCTAssertEqual(r.melFrames, 1000)
        XCTAssertEqual(r.projT, 125)
    }

    func testAbsentManifestFallsBackTo10s() {
        // nil data — the old-asset path (no parakeet.json on disk).
        let r = ParakeetTDTEngine.parseWindowConfig(nil)
        XCTAssertEqual(r.melFrames, 1000)
        XCTAssertEqual(r.projT, 125)
    }

    func testMalformedInputsAllFallBack() {
        let bad: [String] = [
            "not json at all",
            "{}",                                             // empty
            #"{"mel_frames":3000}"#,                          // missing proj_t
            #"{"proj_t":375}"#,                               // missing mel_frames
            #"{"mel_frames":"3000","proj_t":"375"}"#,         // wrong types
            #"{"mel_frames":0,"proj_t":0}"#,                  // non-positive
            #"{"mel_frames":-3000,"proj_t":-375}"#,           // negative
            #"{"mel_frames":3000,"proj_t":100}"#,             // proj_t != mel_frames/8
            #"{"mel_frames":3000,"proj_t":374}"#,             // off-by-one on the invariant
        ]
        for json in bad {
            let r = parse(json)
            XCTAssertEqual(r.melFrames, 1000, "should fall back for: \(json)")
            XCTAssertEqual(r.projT, 125, "should fall back for: \(json)")
        }
    }

    func testEmptyDataFallsBack() {
        let r = ParakeetTDTEngine.parseWindowConfig(Data())
        XCTAssertEqual(r.melFrames, 1000)
        XCTAssertEqual(r.projT, 125)
    }
}
