// The bundled model registry (Resources/models.json) — decodes, and every entry
// obeys the schema invariants. Deterministic, no network, runs in CI always.

import XCTest
@testable import VoixfulEngine

final class ModelRegistryTests: XCTestCase {
    /// The bundled JSON decodes and is non-empty.
    func testBundledDecodes() throws {
        let registry = try ModelRegistry.load()
        XCTAssertFalse(registry.models.isEmpty, "registry should ship models")
        XCTAssertEqual(registry.schemaVersion, 1)
        // Eleven entries: the 4 single-checkpoint backends (nemotron, granite,
        // cohere, canary) + Parakeet v2/v3 + five Whisper variants
        // (base, small, large-v3-turbo, medium, large-v3). Cohere is included
        // but excluded from downloads.
        XCTAssertEqual(registry.models.count, 11)
    }

    /// `downloadDir` is set and expands its leading `~`.
    func testDownloadDirExpands() throws {
        let registry = try ModelRegistry.load()
        XCTAssertTrue(registry.downloadDir.hasPrefix("~"), "default dir is home-relative")
        let expanded = registry.downloadDirURL.path
        XCTAssertFalse(expanded.hasPrefix("~"), "downloadDirURL should expand ~")
        XCTAssertTrue(expanded.contains("Voixful/Models"))
    }

    /// Every `backend` is a valid `ModelBackend` rawValue.
    func testBackendsAreValid() throws {
        let registry = try ModelRegistry.load()
        for model in registry.models {
            XCTAssertNotNil(
                model.modelBackend,
                "\(model.id): backend '\(model.backend)' is not a ModelBackend rawValue"
            )
        }
    }

    /// accuracy and speed are within 0…10.
    func testScoresInRange() throws {
        let registry = try ModelRegistry.load()
        for model in registry.models {
            XCTAssertTrue((0...10).contains(model.accuracy), "\(model.id): accuracy out of range")
            XCTAssertTrue((0...10).contains(model.speed), "\(model.id): speed out of range")
        }
    }

    /// `singlePassWindowSeconds` is null iff `streaming` — and, when present,
    /// matches the backend's `ModelBackend.singlePassWindowSeconds`.
    func testWindowNullIffStreaming() throws {
        let registry = try ModelRegistry.load()
        for model in registry.models {
            if model.streaming {
                XCTAssertNil(model.singlePassWindowSeconds, "\(model.id): streaming ⇒ null window")
            } else {
                XCTAssertNotNil(model.singlePassWindowSeconds, "\(model.id): non-streaming ⇒ window")
            }
            // The window must agree with the core ModelBackend metadata.
            if let backend = model.modelBackend {
                XCTAssertEqual(
                    model.singlePassWindowSeconds,
                    backend.singlePassWindowSeconds,
                    "\(model.id): window disagrees with ModelBackend.\(backend.rawValue)"
                )
            }
        }
    }

    /// Ids are unique.
    func testIdsUnique() throws {
        let registry = try ModelRegistry.load()
        let ids = registry.models.map(\.id)
        XCTAssertEqual(Set(ids).count, ids.count, "duplicate model id(s): \(ids)")
    }

    /// Download repos are non-empty except for explicitly-excluded models.
    func testDownloadReposNonEmpty() throws {
        let registry = try ModelRegistry.load()
        for model in registry.models {
            if model.excludeFromDownloads {
                // Excluded (gated/local-only) models may omit a repo.
                continue
            }
            XCTAssertFalse(
                model.download.repo.isEmpty,
                "\(model.id): downloadable model must have a repo"
            )
            XCTAssertEqual(model.download.source, "huggingface", "\(model.id): unexpected source")
            XCTAssertFalse(model.download.assetName.isEmpty, "\(model.id): missing assetName")
        }
    }

    /// wer, when present, is non-negative; convertedSizeMB and params are set.
    func testMetadataSanity() throws {
        let registry = try ModelRegistry.load()
        for model in registry.models {
            if let wer = model.wer {
                XCTAssertGreaterThan(wer, 0, "\(model.id): WER should be positive")
            }
            XCTAssertGreaterThan(model.convertedSizeMB, 0, "\(model.id): size should be positive")
            XCTAssertFalse(model.params.isEmpty, "\(model.id): params should be set")
            XCTAssertFalse(model.languages.isEmpty, "\(model.id): needs at least one language")
        }
    }

    /// Exactly one model is flagged as the default download.
    func testExactlyOneDefaultDownload() throws {
        let registry = try ModelRegistry.load()
        let defaults = registry.models.filter(\.isDefaultDownload)
        XCTAssertEqual(defaults.count, 1, "expected exactly one default download")
        // Nemotron: native streaming, lowest latency — the sensible default.
        XCTAssertEqual(registry.defaultDownload?.id, "nemotron-speech-streaming-en-0.6b")
    }

    /// The embedded `scoring` buckets are present and non-empty — the data
    /// carries its own derivation rules (mirrored in Resources/README.md).
    func testScoringPresent() throws {
        let registry = try ModelRegistry.load()
        let scoring = try XCTUnwrap(registry.scoring, "registry should embed scoring buckets")
        XCTAssertFalse(scoring.accuracyByWER.isEmpty, "accuracyByWER buckets missing")
        XCTAssertFalse(scoring.speedByPass.isEmpty, "speedByPass buckets missing")
    }

    /// Every `accuracy` is exactly the score the embedded WER buckets derive from
    /// that model's `wer` — accuracy is DERIVED, never hand-picked.
    func testAccuracyMatchesDocumentedBuckets() throws {
        let registry = try ModelRegistry.load()
        let buckets = try XCTUnwrap(registry.scoring?.accuracyByWER)
        // Ascending by maxWER: the tightest bucket the WER fits wins.
        let sorted = buckets.sorted { $0.maxWER < $1.maxWER }
        for model in registry.models {
            guard let wer = model.wer else { continue }
            let expected = sorted.first { wer <= $0.maxWER }?.score
            XCTAssertEqual(
                model.accuracy, expected,
                "\(model.id): accuracy \(model.accuracy) != bucket for WER \(wer) (expected \(expected.map(String.init) ?? "nil"))"
            )
        }
    }

    /// Native-streaming models take the streaming speed bucket's score; that
    /// bucket exists and is the top score (10).
    func testStreamingSpeedMatchesBucket() throws {
        let registry = try ModelRegistry.load()
        let streamingBucket = try XCTUnwrap(
            registry.scoring?.speedByPass.first { $0.tier == "native-streaming" },
            "expected a native-streaming speed bucket"
        )
        for model in registry.models where model.streaming {
            XCTAssertEqual(
                model.speed, streamingBucket.score,
                "\(model.id): streaming model should take the native-streaming speed score"
            )
        }
    }

    /// Every window agrees with the core `ModelBackend` metadata for its backend —
    /// and every backend actually parses (so this is a real check, not a skip).
    func testWindowMatchesBackendForEveryEntry() throws {
        let registry = try ModelRegistry.load()
        for model in registry.models {
            let backend = try XCTUnwrap(
                model.modelBackend,
                "\(model.id): backend '\(model.backend)' is not a ModelBackend rawValue"
            )
            XCTAssertEqual(
                model.singlePassWindowSeconds, backend.singlePassWindowSeconds,
                "\(model.id): window disagrees with ModelBackend.\(backend.rawValue)"
            )
        }
    }

    /// The install asset follows the `<id>-palette4.aimodel` naming the convert
    /// pipelines emit.
    func testAssetNameFollowsConvention() throws {
        let registry = try ModelRegistry.load()
        for model in registry.models {
            XCTAssertEqual(
                model.download.assetName, "\(model.id)-palette4.aimodel",
                "\(model.id): assetName should be '<id>-palette4.aimodel'"
            )
        }
    }

    /// A default download is always resolvable.
    func testDefaultDownloadResolves() throws {
        let registry = try ModelRegistry.load()
        XCTAssertNotNil(registry.defaultDownload, "a default download must be resolvable")
    }

    /// `ModelRegistry.bundled` (the trapping accessor) resolves.
    func testBundledAccessor() {
        XCTAssertFalse(ModelRegistry.bundled.models.isEmpty)
    }
}
