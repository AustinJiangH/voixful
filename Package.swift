// swift-tools-version: 6.0
// Nemotron 3.5 ASR on Core AI — see PLAN.md §6 for the package layout this manifest realizes.

import PackageDescription

let package = Package(
    name: "NemotronASR",
    platforms: [
        // Resolves the Phase A VERIFY: targets macOS 27 (Core AI's SDK floor).
        // Verified against Swift 6.4 / SDK 27.0 on 2026-06-11.
        .macOS("27.0"),
    ],
    products: [
        .library(
            name: "NemotronASR",
            targets: ["NemotronASR"]
        ),
        .executable(
            name: "nemotron-asr",
            targets: ["NemotronASRCLI"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.4.0"),
        // TODO (Phase E): add a SentencePiece dependency for tokenizer.
        // Candidates: a SwiftSentencePiece package, or vendoring the C++ library via a
        // SwiftPM system module. Resolved in PLAN.md §6 / Phase E.
    ],
    targets: [
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
