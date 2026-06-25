# convert/cohere — Cohere Transcribe 2B → Core AI `.aimodel`

Converts **Cohere Transcribe** (`CohereLabs/cohere-transcribe-03-2026`) into a Core AI
`.aimodel`. This is Voixful's first **attention encoder-decoder (AED)** backend —
a different architecture from the Nemotron / Parakeet **transducers**.

## Why this model

- **Largest ASR model we target** — 2B params (~3× Parakeet 0.6B).
- **Different architecture** — cross-attention encoder-decoder, not RNN-T/TDT. This
  exercises everything the transducer runtimes don't: cross-attention, dual KV caches
  (decoder self-attn + cross-attn), autoregressive greedy decode with EOS, and
  byte-fallback BPE detokenization.
- SOTA: #1 on the HF Open ASR leaderboard (~5.4% avg WER), 14 languages.

## Architecture (from the model card + `cohere-transcribe-03-2026-coreml`)

| | value |
|---|---|
| Encoder | Fast-Conformer, **48 layers**, hidden **1280**, full-context |
| Encoder input | mel `[1, 128, 3500]` = fixed **35 s** window (160 hop / 16 kHz) |
| Decoder | autoregressive transformer, **8 layers**, hidden **1024**, 8 heads, head-dim 128 |
| Attention | decoder **self-attention (causal) + cross-attention** over encoder output |
| KV cache | decoder self + cross caches; `maxSeqLen = 108` decode steps |
| Vocab | 16,384 (byte-fallback BPE; CJK via byte fallback) |
| Long-form | sliding window, 35 s window / 5 s overlap, token-level merge (LCS) |
| Params | ~2B, **>90% in the encoder** (decoder is lightweight → cheap autoregression) |

## Planned `.aimodel` entrypoints

Mirrors the per-stage split the runtime needs (and what FluidAudio's CoreML pack uses):

- `encode(mel[1,128,3500], mel_length[1]) -> enc_out[1, T_enc, 1280]`
- `decode_step(prev_token[1,1], <self-attn KV state>, <cross-attn KV>) -> logits[1,1,16384]`
  - cross-attn K/V are computed **once** from `enc_out` and cached; self-attn KV grows per step.

The open question is whether `coreai_torch 0.4.0` can express the decoder's **dual
stateful KV cache** (self-attn cache mutated per step + cross-attn cache written once)
as Core AI `MutableBuffers`. That is the make-or-break feasibility gate — settle it
with a minimal decoder export before building the full runtime.

## Memory plan (2B is the real test)

Core AI specializes FP16 → FP32 at load, so a 2B FP16 asset would be ~15 GB RSS —
infeasible. **Palettize the encoder from the start** (4-bit ≈ ~470 MB for the encoder)
the same way the Nemotron / Parakeet pipelines do. The decoder is small (~0.3 GB).

## Status

**Not started — blocked on model access.** `CohereLabs/cohere-transcribe-03-2026` is
**gated** (license acceptance + HF auth) and ships custom modeling code
(`modeling_cohere_asr.py`, loaded via `trust_remote_code=True`). To begin:

1. Accept the model license on its Hugging Face page.
2. Authenticate locally (`hf auth login`, or set `HF_TOKEN`).
3. Then: study `modeling_cohere_asr.py` for the exact encoder/decoder module + cache
   interface, write the wrapper modules, and run the decoder feasibility export.

## License

- Conversion code → Apache-2.0 (repo root `LICENSE` / `NOTICE`).
- `.aimodel` weights → derived from `CohereLabs/cohere-transcribe-03-2026`; retain its
  upstream license + attribution when redistributing.
