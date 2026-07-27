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
        .library(name: "VoixfulCanaryQwen", targets: ["VoixfulCanaryQwen"]),
        .library(name: "VoixfulWhisper", targets: ["VoixfulWhisper"]),
        .library(name: "VoixfulQwen3ASR", targets: ["VoixfulQwen3ASR"]),
        // Shared full-context (batch) transcriber lifecycle (Parakeet + Cohere).
        .library(name: "VoixfulFullContext", targets: ["VoixfulFullContext"]),
        // Umbrella: imports every backend; holds the BackendRegistry table + the
        // ResidentModel.load(...) entry point.
        .library(name: "VoixfulKit", targets: ["VoixfulKit"]),
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
        .target(
            name: "VoixfulCanaryQwen",
            dependencies: ["VoixfulSpeech", "VoixfulEngine", "VoixfulFullContext"]
        ),
        .target(
            name: "VoixfulWhisper",
            dependencies: ["VoixfulSpeech", "VoixfulEngine", "VoixfulFullContext"]
        ),
        .target(
            name: "VoixfulQwen3ASR",
            dependencies: ["VoixfulSpeech", "VoixfulEngine", "VoixfulFullContext"]
        ),

        // MARK: Umbrella — every backend + the registry table + ResidentModel.load
        .target(
            name: "VoixfulKit",
            dependencies: [
                "VoixfulSpeech",
                "VoixfulEngine",
                "VoixfulFullContext",
                "VoixfulNemotronStreaming",
                "VoixfulParakeetTDT",
                "VoixfulCohereTranscribe",
                "VoixfulGraniteNAR",
                "VoixfulCanaryQwen",
                "VoixfulWhisper",
                "VoixfulQwen3ASR",
            ]
        ),

        // MARK: Brand CLI
        .executableTarget(
            name: "VoixfulCLI",
            dependencies: [
                "VoixfulSpeech",
                "VoixfulTUI",
                "VoixfulKit",
                "VoixfulNemotronStreaming",
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
            dependencies: [],
            // The model registry catalog (data, not Swift) — bundled so apps
            // depending on the package read it via `Bundle.module` /
            // `ModelRegistry.bundled`. Copied verbatim so it stays human-editable.
            resources: [
                .copy("Resources/models.json"),
            ]
            // NOTE: we intentionally do NOT pass `-Xcc -DACCELERATE_NEW_LAPACK=1`
            // via `.unsafeFlags` here. SwiftPM forbids resolving a package by
            // version tag if any depended-on product uses unsafe flags, which
            // would block `.package(url:, from: "x.y.z")`. The flag only silenced
            // Accelerate CBLAS deprecation warnings; dropping it keeps the package
            // version-consumable. (Follow-up: move the cblas_* calls behind a tiny
            // C shim target that sets the macro via the safe `.define` cSetting.)
        ),
        .testTarget(
            name: "VoixfulEngineTests",
            dependencies: ["VoixfulEngine", "VoixfulFullContext", "VoixfulNemotronStreaming", "VoixfulGraniteNAR", "VoixfulWhisper", "VoixfulSpeech", "VoixfulKit"]
        ),
    ]
)
