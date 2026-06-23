// swift-tools-version: 6.0
// Voixful — SpeechAnalyzer-shaped on-device speech API over local models.
// Monorepo: VoixfulSpeech (brand core) + per-model backends (VoixfulNemotron).
// See DESIGN_ANALYZER_API.md for the topology this manifest realizes.

import PackageDescription

let package = Package(
    name: "Voixful",
    platforms: [
        .macOS("27.0"),
    ],
    products: [
        // Brand core — the model-agnostic public contract.
        .library(name: "VoixfulSpeech", targets: ["VoixfulSpeech"]),
        // Nemotron backend — one model behind the brand surface.
        .library(name: "VoixfulNemotron", targets: ["VoixfulNemotron"]),
        // Brand CLI: `voixful transcribe …`.
        .executable(name: "voixful", targets: ["VoixfulCLI"]),

        // Existing engine library + legacy CLI (mic/file), kept during migration.
        .library(name: "NemotronASR", targets: ["NemotronASR"]),
        .executable(name: "nemotron-asr", targets: ["NemotronASRCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.4.0"),
    ],
    targets: [
        // MARK: Brand core (no model dependency)
        .target(
            name: "VoixfulSpeech",
            dependencies: []
        ),

        // MARK: Nemotron backend (depends on core + the engine)
        .target(
            name: "VoixfulNemotron",
            dependencies: ["VoixfulSpeech", "NemotronASR"]
        ),

        // MARK: Brand CLI
        .executableTarget(
            name: "VoixfulCLI",
            dependencies: [
                "VoixfulSpeech",
                "VoixfulNemotron",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),

        // MARK: Existing engine + legacy CLI + tests
        .target(
            name: "NemotronASR",
            dependencies: []
        ),
        .executableTarget(
            name: "NemotronASRCLI",
            dependencies: [
                "NemotronASR",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(
            name: "NemotronASRTests",
            dependencies: ["NemotronASR"]
        ),
    ]
)
