# convert/cohere-transcribe — Cohere Transcribe 2B → Core AI `.aimodel`

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

## `.aimodel` entrypoints (validated against the upstream CoreML pack)

The exact I/O was read from FluidInference's ungated CoreML mirror (`model.mil`),
so these shapes are confirmed, not guessed:

```
encode(input_features[1,128,3500] fp32, feature_length[1] i32)
    -> hidden_states[1, 438, 1024]          # 3500 mel frames / 8 subsample = 438; proj 1280->1024

decode_step(                                 # external cache, STATELESS
    input_id[1,1] i32, position_id[1,1] i32,
    encoder_hidden_states[1,438,1024] fp32,
    attention_mask[1,1,1,108] fp32,          # causal self-attn mask (upstream flexes 1..108; we fix 108)
    cross_attention_mask[1,1,1,438] fp32,    # masks padded encoder frames
    k_cache_0..7[1,8,108,128], v_cache_0..7[1,8,108,128]   # 8 layers, 8 heads, maxlen 108, head-dim 128
) -> logits[1,16384], k_cache_0..7_out, v_cache_0..7_out
```

**Feasibility gate — resolved.** The upstream design uses an **external KV cache**
(caches are plain tensor inputs/outputs, not internal state) and caches **only
self-attention**; cross-attention K/V are recomputed each step from
`encoder_hidden_states`. So both functions are **stateless** — no Core AI
`MutableBuffers` required, just two fixed-shape `coreai_torch` exports (simpler than
Parakeet's stateful LSTM). The Swift runtime owns the 16 cache tensors and threads
them step to step. The only adaptation vs upstream: `coreai_torch 0.4.0` can't carry
a dynamic axis, so we fix `attention_mask` to `[1,1,1,108]` and rely on mask values
(not shape) to hide future positions.

## Memory plan (2B is the real test)

Core AI specializes FP16 → FP32 at load, so a 2B FP16 asset would be ~15 GB RSS —
infeasible. **Palettize the encoder from the start** (4-bit ≈ ~470 MB for the encoder)
the same way the Nemotron / Parakeet pipelines do. The decoder is small (~0.3 GB).

## Status

**Works end-to-end.** The converter emits a 2-function `.aimodel`
(`encode` + `decode_step`), and `NemotronASR.CohereEngine` runs it on-device:

```sh
voixful transcribe clip.wav --model cohere-palette4.aimodel
```

FP16 and palette4 both produce torch-identical transcripts; palettizing the encoder
(>90% of the params) cuts the asset 7.6 GB → 1.5 GB and peak RSS 15.4 GB → ~3 GB.

### Conversion gotcha — the `coreai_torch` broadcast bug

The decoder derailed (correct for ~15 tokens, then a greedy loop) until we traced it
to a real `coreai_torch 0.4.0` bug: it **mis-converts a broadcast multiply across a
size-1 dim**. The external-cache write `cache*(1-sel) + new*sel` broadcasts the new
`[1,H,1,D]` K/V against the `[1,H,M,D]` cache; under conversion the new value leaked
into *every* cache slot (logits off by ~10, enough to flip borderline greedy tokens).
Fix: materialize both operands to `[1,H,M,D]` with `.repeat(...)` (a real copy) before
the multiply — never a stride-0 broadcast view. Bisection that found it: embedding
bit-exact → QKV projection bit-exact → cache-write off by 5.

Two more decoder specifics: the wrapper spells out attention manually (not
`F.scaled_dot_product_attention`) and emits raw logits (skips the head's
`log_softmax`, argmax-identical) — both keep the conversion clean.

## License

- Conversion code → Apache-2.0 (repo root `LICENSE` / `NOTICE`).
- `.aimodel` weights → derived from `CohereLabs/cohere-transcribe-03-2026`; retain its
  upstream license + attribution when redistributing.
