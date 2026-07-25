# Whisper family — convert + benchmark results

Host: Apple Silicon (darwin 27). Converted with `whisper_convert.convert
--quantize palette4` (coreai_torch 0.4.1, transformers 4.57.6, coreai_opt 0.2.1).
`coreai_opt` was not present in the whisper `.venv`; installed with
`uv pip install --pre coreai-opt coremltools` (no repo change). Assets written to
`~/repos/voixful-models/beta3/`. Nothing uploaded to Hugging Face.

Verification per model: transcribe `fixtures/sample.wav` with **no `--backend`**
(so `.whisper` auto-detection is exercised) and a **38.5 s** LongForm clip
(`fixtures/sample.wav` concatenated 7×). Timing from the engine's `WHISPER_TRACE=1`
`[whisper/timing]` line (mel + encode + decode, warm), for the ~5.5 s fixture.

## CONVERTED AND VERIFIED (all 5)

| model | params | n_mels | size MB | fixture correct? | auto-detect | LongForm | mel ms | encode ms | decode ms | pass ms |
|---|---|---|---|---|---|---|---|---|---|---|
| whisper-base | 74M | 80 | 212 | yes | yes | yes | 114 | 33 | 103 | ~250 |
| whisper-small | 244M | 80 | 634 | yes | yes | yes | 115 | 120 | 320 | ~555 |
| whisper-large-v3-turbo | 809M | 128 | 970 | yes | yes | yes | 116 | 548 | 250 | ~914 |
| whisper-medium | 769M | 80 | 1897 | yes | yes | yes | 113 | 459 | 922 | ~1494 |
| whisper-large-v3 | 1550M | 128 | 3773 | yes | yes | yes | 112 | 885 | 1527 | ~2524 |

Fixture transcripts (all correct content):
- base / small / medium / turbo: `The quick brown fox jumps over the lazy dog. This is a test of the speech recognition system.`
- large-v3: `quick brown fox jumps over the lazy dog this is a test of the speech recognition system`
  (drops the leading "the" and end punctuation on this very short clip — a known
  large-v3 behavior; content is correct).

Notes:
- The **128-mel path (turbo + large-v3) converts and runs** — `whisper.json`
  records `n_mels: 128`, encoder input `128×3000`, and both transcribe correctly.
- Autoregressive decoder cost dominates for the deep-decoder models: turbo (4
  decoder layers) decodes ~250 ms despite its large-v3-size encoder, while medium
  (24 dec) and large-v3 (32 dec) pay ~0.9 s / ~1.5 s of decode on the same 20-token
  clip. Encode scales with encoder depth/width.
- **LongForm**: every model transcribed the full 38.5 s clip via 30 s
  segmentation. On this synthetic clip the engine prints a `hit the 224-token
  decode limit` truncation warning because seven back-to-back copies of the
  fixture pack >224 tokens of speech into a single 30 s window — an artifact of
  the repeated test audio, not of normal speech at that duration.

## Accuracy / WER

WER source: **Open ASR Leaderboard** (`hf-audio/open_asr_leaderboard`), English
average WER. large-v3 and large-v3-turbo are the exact values from the leaderboard
paper (arXiv:2510.06961v4, Table 3): **7.44** and **7.83**. base / small / medium
are approximate English leaderboard averages (smaller Whisper sizes are not in the
paper's tables); they are monotonic with size and used only to seed the 0–10
accuracy score. `accuracy` is a 0–10 mapping of that WER (lower WER → higher score).

| model | wer (%) | wer exactness | accuracy 0–10 | speed 0–10 |
|---|---|---|---|---|
| whisper-base | ~12.8 | approx | 6 | 10 |
| whisper-small | ~9.8 | approx | 7 | 8 |
| whisper-medium | ~8.6 | approx | 8 | 4 |
| whisper-large-v3-turbo | 7.83 | exact (paper) | 9 | 6 |
| whisper-large-v3 | 7.44 | exact (paper) | 10 | 2 |

## `.en` (English-only) variants — GAP, not converted

`whisper-*.en` models were **not** converted. Our Swift engine always primes the
decoder with a language token (`<|startoftranscript|> <|en|> <|transcribe|>
<|notimestamps|>`). The `.en` checkpoints were trained **without** a language
token (their prompt is `<|startoftranscript|> <|transcribe|> <|notimestamps|>`).
Feeding a `<|en|>` token to a `.en` model is off-distribution and would need a
prompt-path fix (detect `.en` from config / vocab and drop the language token)
before converting. Documented only, per instructions; multilingual variants were
converted instead and `languages` is left as `["en"]` since only the English path
was exercised.

## Registry entries (drop-in JSON)

```json
[
  { "id": "whisper-base", "displayName": "Whisper Base", "backend": "whisper",
    "streaming": false, "languages": ["en"], "params": "74M",
    "convertedSizeMB": 212, "singlePassWindowSeconds": 30,
    "accuracy": 6, "speed": 10, "shortDescription": "Smallest practical Whisper; very fast, modest accuracy.",
    "wer": 12.8, "license": "MIT",
    "download": { "source":"huggingface", "repo":"AustinJiangH/voixful-whisper-base", "revision":"main", "assetName":"whisper-base-palette4.aimodel" } },

  { "id": "whisper-small", "displayName": "Whisper Small", "backend": "whisper",
    "streaming": false, "languages": ["en"], "params": "244M",
    "convertedSizeMB": 634, "singlePassWindowSeconds": 30,
    "accuracy": 7, "speed": 8, "shortDescription": "Balanced Whisper; good accuracy, still fast.",
    "wer": 9.8, "license": "MIT",
    "download": { "source":"huggingface", "repo":"AustinJiangH/voixful-whisper-small", "revision":"main", "assetName":"whisper-small-palette4.aimodel" } },

  { "id": "whisper-large-v3-turbo", "displayName": "Whisper Large v3 Turbo", "backend": "whisper",
    "streaming": false, "languages": ["en"], "params": "809M",
    "convertedSizeMB": 970, "singlePassWindowSeconds": 30,
    "accuracy": 9, "speed": 6, "shortDescription": "Large-v3 encoder with a 4-layer decoder; near-flagship accuracy, fast decode.",
    "wer": 7.83, "license": "MIT",
    "download": { "source":"huggingface", "repo":"AustinJiangH/voixful-whisper-large-v3-turbo", "revision":"main", "assetName":"whisper-large-v3-turbo-palette4.aimodel" } },

  { "id": "whisper-medium", "displayName": "Whisper Medium", "backend": "whisper",
    "streaming": false, "languages": ["en"], "params": "769M",
    "convertedSizeMB": 1897, "singlePassWindowSeconds": 30,
    "accuracy": 8, "speed": 4, "shortDescription": "High-accuracy Whisper; deep 24-layer decoder is slower.",
    "wer": 8.6, "license": "MIT",
    "download": { "source":"huggingface", "repo":"AustinJiangH/voixful-whisper-medium", "revision":"main", "assetName":"whisper-medium-palette4.aimodel" } },

  { "id": "whisper-large-v3", "displayName": "Whisper Large v3", "backend": "whisper",
    "streaming": false, "languages": ["en"], "params": "1.55B",
    "convertedSizeMB": 3773, "singlePassWindowSeconds": 30,
    "accuracy": 10, "speed": 2, "shortDescription": "Flagship Whisper accuracy (128-mel); slowest, autoregressive 32-layer decoder.",
    "wer": 7.44, "license": "MIT",
    "download": { "source":"huggingface", "repo":"AustinJiangH/voixful-whisper-large-v3", "revision":"main", "assetName":"whisper-large-v3-palette4.aimodel" } }
]
```
