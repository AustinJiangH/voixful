# Voixful model registry (`models.json`)

A machine-readable catalog of every shipping model + its metadata. It is **data,
not Swift** — bundled as a SwiftPM resource so any app depending on the package
can list/download models via `ModelRegistry.bundled` (see
`Sources/VoixfulEngine/Shared/ModelRegistry.swift`). Edit this JSON to change the
catalog; nothing is hard-coded.

## Top level

| Field | Meaning |
|---|---|
| `schemaVersion` | Integer; bump on breaking schema changes. |
| `downloadDir` | Default install directory for downloaded `.aimodel`s. `~` expands to the user's home (`ModelRegistry.downloadDirURL`). |
| `scoring` | The documented `accuracy` / `speed` bucketing (below), embedded so it travels with the data. |
| `models` | Array of model entries. |

## Per-model schema

`id` (stable slug) · `displayName` · `backend` (must equal a `ModelBackend`
rawValue: `nemotron` \| `parakeet` \| `cohere` \| `granite` \| `canary` \|
`whisper`) · `architecture` · `upstreamRepo` (attribution) · `streaming` (bool) ·
`languages` (BCP-47 codes) · `params` (e.g. `"0.6B"`) · `convertedSizeMB` (int,
the palette4 `.aimodel` size on disk) · `singlePassWindowSeconds` (number, or
`null` for a native-streaming backend) · `accuracy` (0–10 int) · `speed` (0–10
int) · `shortDescription` · `wer` (number or `null`, Open ASR Leaderboard mean) ·
`license` (SPDX-ish) · `excludeFromDownloads` (bool) · optional
`isDefaultDownload` (bool) · optional `notes` (estimate/caveat) · `download`
(`source` = `"huggingface"`, `repo`, `revision`, `assetName`).

Invariant: `singlePassWindowSeconds` is `null` **iff** `streaming` is `true`
(only Nemotron streams natively; every full-context backend has a window). The
window value is sourced from `ModelBackend.singlePassWindowSeconds`.

## Scoring — `accuracy` and `speed` (0–10)

Both scores are **derived from documented buckets**, not hand-picked, so they are
reproducible. The buckets are also embedded under the top-level `scoring` key.

### `accuracy` — bucketed from Open ASR Leaderboard mean WER (lower WER → higher)

| WER (mean, ≤) | accuracy |
|---|---|
| 5.0 | 9 |
| 5.5 | 8 |
| 6.0 | 7 |
| 7.0 | 6 |
| 8.0 | 5 |
| 10.0 | 4 |
| (worse / unknown) | 3 |

Applied: granite 4.95 → 9 · canary 5.06 → 8 · cohere 5.20 → 8 · parakeet-v2
5.39 → 8 · parakeet-v3 5.66 → 7 · nemotron 5.73 → 7 · whisper-large-v3 7.44 → 5 ·
whisper-large-v3-turbo 7.83 → 5 · whisper-medium 8.6 → 4 · whisper-small
9.8 → 4 · whisper-base.en 10.33 → 3 · whisper-base 12.8 → 3.

### `speed` — bucketed from native-streaming vs. full-context single-pass latency

Latencies are the warm per-utterance pass numbers measured on M5 Pro (README
"Performance knobs & measured baselines").

| Condition | speed |
|---|---|
| native streaming (partials while you speak) | 10 |
| full-context pass ≤ 0.1 s | 9 |
| ≤ 0.5 s | 7 |
| ≤ 1.5 s | 5 |
| ≤ 3.0 s | 3 |
| slower | 2 |

Applied: nemotron (streaming) → 10 · parakeet v2/v3 (~0.07 s) → 9 · granite /
cohere (~0.29 s) → 7 · whisper-base / whisper-base.en (~0.25 s) → 7 · whisper-small (~0.55 s) → 5 ·
whisper-large-v3-turbo (~0.9 s) → 5 · whisper-medium (~1.5 s) → 5 ·
whisper-large-v3 (~2.5 s) → 3 · canary (~4 s) → 2.

## Estimated vs. measured

`convertedSizeMB` is **measured** (`du -sm` of the published palette4 asset).
Canary and Granite TurboCTC were measured on their release-toolchain builds
(coreai-core 1.0.0b3; Canary's fp16-KV build is larger than the earlier beta
build). Every other entry is still the beta build's size, which the release
rebuilds match to within ~10 MB. Cohere is measured locally only (not
published). `wer`
values are Open ASR Leaderboard means; the Whisper figures are the published
English WERs for each checkpoint.
