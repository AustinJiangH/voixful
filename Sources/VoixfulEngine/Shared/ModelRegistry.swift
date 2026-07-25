// The model registry — a Codable view over the bundled `Resources/models.json`
// catalog. This type is DATA-BACKED: it decodes the JSON resource rather than
// hard-coding the catalog in Swift, so apps depending on the package get the
// list (and can download the assets) from a single editable source of truth.
//
// See Resources/models.json + Resources/README.md for the schema and the
// documented accuracy/speed scoring. Dependency-free and Sendable.

import Foundation

/// The decoded model catalog: a default download directory + every shipping
/// model's metadata. Load the bundled copy with `ModelRegistry.bundled`.
public struct ModelRegistry: Codable, Sendable, Equatable {
    /// Schema version; bump on breaking changes to the JSON shape.
    public let schemaVersion: Int
    /// Default install directory for downloaded `.aimodel`s, as written in the
    /// JSON (may start with `~`). Use `downloadDirURL` for an expanded path.
    public let downloadDir: String
    /// The documented accuracy/speed bucketing, carried alongside the data.
    public let scoring: Scoring?
    /// Every shipping model.
    public let models: [ModelEntry]

    /// `downloadDir` with a leading `~` expanded to the user's home directory.
    public var downloadDirURL: URL {
        URL(fileURLWithPath: (downloadDir as NSString).expandingTildeInPath)
    }

    /// Look up a model by its stable `id`.
    public func model(id: String) -> ModelEntry? { models.first { $0.id == id } }

    /// Models that should be offered for auto-download (excludes gated/local-only
    /// entries such as Cohere).
    public var downloadableModels: [ModelEntry] { models.filter { !$0.excludeFromDownloads } }

    /// The model marked `isDefaultDownload`, if any.
    public var defaultDownload: ModelEntry? { models.first { $0.isDefaultDownload } }
}

/// One installable model checkpoint + its display / install / scoring metadata.
public struct ModelEntry: Codable, Sendable, Equatable, Identifiable {
    /// Stable slug (also the catalog key).
    public let id: String
    public let displayName: String
    /// Runtime backend — equals a `ModelBackend` rawValue.
    public let backend: String
    /// Architecture family (human-readable).
    public let architecture: String?
    /// Upstream checkpoint the weights derive from (attribution).
    public let upstreamRepo: String?
    /// Native streaming (words appear while you speak) vs. re-transcribe partials.
    public let streaming: Bool
    /// Supported languages as BCP-47 codes.
    public let languages: [String]
    /// Parameter count, e.g. "0.6B".
    public let params: String
    /// Size on disk of the palette4 `.aimodel` (MB).
    public let convertedSizeMB: Int
    /// Longest audio (seconds) transcribed in one pass; `nil` for streaming.
    public let singlePassWindowSeconds: Double?
    /// Accuracy score, 0…10 (bucketed from WER — see Resources/README.md).
    public let accuracy: Int
    /// Speed score, 0…10 (bucketed from pass latency — see Resources/README.md).
    public let speed: Int
    /// One-line "what it's best for".
    public let shortDescription: String
    /// Open ASR Leaderboard mean WER for the source checkpoint (lower is better).
    public let wer: Double?
    /// SPDX-ish weights license identifier.
    public let license: String
    /// Whether to exclude from auto-download (e.g. gated upstream).
    public let excludeFromDownloads: Bool
    /// Whether this is the default download for its family. Defaults to `false`.
    public let isDefaultDownload: Bool
    /// Optional estimate/caveat note.
    public let notes: String?
    /// Where to fetch the asset from.
    public let download: Download

    /// The parsed `ModelBackend`, or `nil` if the JSON names an unknown backend.
    public var modelBackend: ModelBackend? { ModelBackend(rawValue: backend) }

    public struct Download: Codable, Sendable, Equatable {
        /// Download source kind; currently always "huggingface".
        public let source: String
        /// Hugging Face repo (`org/name`); empty for excluded/local-only models.
        public let repo: String
        /// Git revision / tag to pin.
        public let revision: String
        /// The `.aimodel` directory name once installed.
        public let assetName: String
    }

    private enum CodingKeys: String, CodingKey {
        case id, displayName, backend, architecture, upstreamRepo, streaming,
             languages, params, convertedSizeMB, singlePassWindowSeconds,
             accuracy, speed, shortDescription, wer, license,
             excludeFromDownloads, isDefaultDownload, notes, download
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        displayName = try c.decode(String.self, forKey: .displayName)
        backend = try c.decode(String.self, forKey: .backend)
        architecture = try c.decodeIfPresent(String.self, forKey: .architecture)
        upstreamRepo = try c.decodeIfPresent(String.self, forKey: .upstreamRepo)
        streaming = try c.decode(Bool.self, forKey: .streaming)
        languages = try c.decode([String].self, forKey: .languages)
        params = try c.decode(String.self, forKey: .params)
        convertedSizeMB = try c.decode(Int.self, forKey: .convertedSizeMB)
        singlePassWindowSeconds = try c.decodeIfPresent(Double.self, forKey: .singlePassWindowSeconds)
        accuracy = try c.decode(Int.self, forKey: .accuracy)
        speed = try c.decode(Int.self, forKey: .speed)
        shortDescription = try c.decode(String.self, forKey: .shortDescription)
        wer = try c.decodeIfPresent(Double.self, forKey: .wer)
        license = try c.decode(String.self, forKey: .license)
        excludeFromDownloads = try c.decode(Bool.self, forKey: .excludeFromDownloads)
        isDefaultDownload = try c.decodeIfPresent(Bool.self, forKey: .isDefaultDownload) ?? false
        notes = try c.decodeIfPresent(String.self, forKey: .notes)
        download = try c.decode(Download.self, forKey: .download)
    }
}

/// The documented accuracy/speed bucketing, mirrored from `Resources/README.md`.
public struct Scoring: Codable, Sendable, Equatable {
    public let note: String
    public let accuracyByWER: [AccuracyBucket]
    public let speedByPass: [SpeedBucket]

    public struct AccuracyBucket: Codable, Sendable, Equatable {
        public let maxWER: Double
        public let score: Int
    }
    public struct SpeedBucket: Codable, Sendable, Equatable {
        public let tier: String?
        public let maxPassSeconds: Double?
        public let score: Int
    }
}

extension ModelRegistry {
    /// Errors surfaced while locating/decoding the bundled catalog.
    public enum RegistryError: Error, CustomStringConvertible {
        case resourceMissing
        case decodeFailed(Error)

        public var description: String {
            switch self {
            case .resourceMissing:
                return "models.json not found in Bundle.module"
            case .decodeFailed(let error):
                return "failed to decode models.json: \(error)"
            }
        }
    }

    /// The catalog bundled with the package (`Resources/models.json`). Traps only
    /// if the resource is malformed — a programmer error, since it ships with the
    /// package. Use `load()` for a throwing variant.
    public static var bundled: ModelRegistry {
        do {
            return try load()
        } catch {
            fatalError("ModelRegistry.bundled: \(error)")
        }
    }

    /// Decode the bundled `Resources/models.json`.
    public static func load() throws -> ModelRegistry {
        guard let url = Bundle.module.url(forResource: "models", withExtension: "json")
                ?? Bundle.module.url(forResource: "models", withExtension: "json", subdirectory: "Resources")
        else {
            throw RegistryError.resourceMissing
        }
        return try load(from: url)
    }

    /// Decode a registry from an arbitrary JSON file (used by tests / tooling).
    public static func load(from url: URL) throws -> ModelRegistry {
        let data = try Data(contentsOf: url)
        do {
            return try JSONDecoder().decode(ModelRegistry.self, from: data)
        } catch {
            throw RegistryError.decodeFailed(error)
        }
    }
}
