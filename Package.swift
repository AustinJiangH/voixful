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
        // Model-agnostic live terminal UI (mic session) — shared by every backend's CLI.
        .library(name: "VoixfulTUI", targets: ["VoixfulTUI"]),
        // Nemotron backend — one model behind the brand surface.
        .library(name: "VoixfulNemotron", targets: ["VoixfulNemotron"]),
        // Brand CLI: `voixful transcribe …`.
        .executable(name: "voixful", targets: ["VoixfulCLI"]),

        // Core AI engine layer — shared runtimes for every model (Nemotron RNN-T,
        // Parakeet TDT, Cohere AED) plus shared mel / tokenizer / NDArray infra.
        .library(name: "VoixfulEngine", targets: ["VoixfulEngine"]),
        // Legacy Nemotron-only CLI (file/mic/languages/smoke), kept during migration.
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

        // MARK: Model-agnostic live terminal UI (depends only on core)
        .target(
            name: "VoixfulTUI",
            dependencies: ["VoixfulSpeech"]
        ),

        // MARK: Nemotron backend (depends on core + the engine)
        .target(
            name: "VoixfulNemotron",
            dependencies: ["VoixfulSpeech", "VoixfulEngine"]
        ),

        // MARK: Brand CLI
        .executableTarget(
            name: "VoixfulCLI",
            dependencies: [
                "VoixfulSpeech",
                "VoixfulTUI",
                "VoixfulNemotron",
                "VoixfulEngine",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            // Embed an Info.plist (with NSMicrophoneUsageDescription) into the
            // executable so macOS TCC grants REAL microphone audio to `voixful
            // mic` — without it a bare CLI binary is fed digital silence. Path
            // is resolved by the linker relative to the package root.
            linkerSettings: [
                .unsafeFlags([
                    "-Xlinker", "-sectcreate",
                    "-Xlinker", "__TEXT",
                    "-Xlinker", "__info_plist",
                    "-Xlinker", "Sources/VoixfulCLI/Info.plist",
                ])
            ]
        ),

        // MARK: Core AI engine (all model runtimes) + legacy CLI + tests
        .target(
            name: "VoixfulEngine",
            dependencies: []
        ),
        .executableTarget(
            name: "NemotronASRCLI",
            dependencies: [
                "VoixfulEngine",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(
            name: "VoixfulEngineTests",
            dependencies: ["VoixfulEngine"]
        ),
    ]
)
