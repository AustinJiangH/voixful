// swift-tools-version: 6.0
// Voixful — SpeechAnalyzer-shaped on-device speech API over local models.
// Monorepo: VoixfulSpeech (brand core) + per-model backends (VoixfulNemotronStreaming).
// See README.md for the topology this manifest realizes.

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
        // Per-model backends — each a `Transcribing` conformer behind the façade.
        .library(name: "VoixfulNemotronStreaming", targets: ["VoixfulNemotronStreaming"]),
        .library(name: "VoixfulParakeetTDT", targets: ["VoixfulParakeetTDT"]),
        .library(name: "VoixfulCohereTranscribe", targets: ["VoixfulCohereTranscribe"]),
        .library(name: "VoixfulGraniteNAR", targets: ["VoixfulGraniteNAR"]),
        // Shared full-context (batch) transcriber lifecycle (Parakeet + Cohere).
        .library(name: "VoixfulFullContext", targets: ["VoixfulFullContext"]),
        // Brand CLI: `voixful transcribe …`.
        .executable(name: "voixful", targets: ["VoixfulCLI"]),

        // Core AI engine layer — shared runtimes for every model (Nemotron RNN-T,
        // Parakeet TDT, Cohere AED) plus shared mel / tokenizer / NDArray infra.
        .library(name: "VoixfulEngine", targets: ["VoixfulEngine"]),
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

        // MARK: Shared full-context (batch) transcriber — the lifecycle Parakeet
        // and Cohere share; generic over the engine.
        .target(
            name: "VoixfulFullContext",
            dependencies: ["VoixfulSpeech", "VoixfulEngine"]
        ),

        // MARK: Per-model backends (each: core façade + the engine)
        .target(
            name: "VoixfulNemotronStreaming",
            dependencies: ["VoixfulSpeech", "VoixfulEngine"]
        ),
        .target(
            name: "VoixfulParakeetTDT",
            dependencies: ["VoixfulSpeech", "VoixfulEngine", "VoixfulFullContext"]
        ),
        .target(
            name: "VoixfulCohereTranscribe",
            dependencies: ["VoixfulSpeech", "VoixfulEngine", "VoixfulFullContext"]
        ),
        .target(
            name: "VoixfulGraniteNAR",
            dependencies: ["VoixfulSpeech", "VoixfulEngine", "VoixfulFullContext"]
        ),

        // MARK: Brand CLI
        .executableTarget(
            name: "VoixfulCLI",
            dependencies: [
                "VoixfulSpeech",
                "VoixfulTUI",
                "VoixfulNemotronStreaming",
                "VoixfulParakeetTDT",
                "VoixfulCohereTranscribe",
                "VoixfulGraniteNAR",
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

        // MARK: Core AI engine (all model runtimes) + tests
        .target(
            name: "VoixfulEngine",
            dependencies: []
        ),
        .testTarget(
            name: "VoixfulEngineTests",
            dependencies: ["VoixfulEngine", "VoixfulFullContext", "VoixfulGraniteNAR", "VoixfulSpeech"]
        ),
    ]
)
