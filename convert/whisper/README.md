# convert/whisper — OpenAI Whisper → Core AI `.aimodel`

Converts **OpenAI Whisper** (default `openai/whisper-base`) into a Core AI
`.aimodel`. Whisper is an **attention encoder-decoder (AED)** — the same shape as
Cohere Transcribe — so this pipeline mirrors `convert/cohere-transcribe/`.

## Why this model

- The most widely-deployed open ASR family; a good coverage anchor.
- **AED**, like Cohere: cross-attention encoder-decoder, autoregressive greedy
  decode with an EOT stop, byte-level BPE detokenization.
- Whisper-specific bring-up work that the Cohere path didn't need:
  - a **fixed 30 s** input (always pad/trim → 3000 log-mel frames → 1500 encoder
    frames), so the single-pass window is inherently 30 s;
  - Whisper's own **log-mel** front-end (log10 + global `(x+4)/4`, no
    pre-emphasis; `n_fft=400`); and
  - a **byte-level BPE** (GPT-2/tiktoken) tokenizer — NOT SentencePiece.

## Architecture (whisper-base; read from `model.config` at convert time)

| | value |
|---|---|
| Encoder | conv1/conv2 frontend + transformer, full-context |
| Encoder input | mel `[1, n_mels, 3000]` = fixed **30 s** window (160 hop / 16 kHz) |
| `n_mels` | **80** (tiny/base/small/medium/large-v1/v2) · **128** (large-v3) |
| Encoder output | `[1, 1500, d]` (3000 mel frames / 2 conv-stride subsample) |
| Decoder | autoregressive transformer, self-attn (causal) + cross-attn |
| Vocab | 51,865 (multilingual) — byte-level BPE |
| Special prime | `<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>` → text → `<|endoftext|>` |

## `.aimodel` entrypoints

The decode entrypoint is deliberately named **`whisper_decode`** (not Cohere's
`decode_step*`) so `ModelBackend.classify` distinguishes Whisper from Cohere; the
Swift rule is checked BEFORE the generic `decode_*` rule.

```
whisper_encode(input_features[1, n_mels, 3000] fp32)
    -> encoder_hidden_states[1, 1500, d]

whisper_decode(                              # external self-attn cache, STATELESS
    input_id[1,1] i32, position_id[1,1] i32,
    encoder_hidden_states[1, 1500, d] fp32,
    self_attention_mask[1,1,1,M] fp32,       # causal mask; M = --max-decode-tokens (224)
    k_cache[L,1,H,M,D], v_cache[L,1,H,M,D]
) -> logits[1,1,vocab], k_cache_out, v_cache_out
```

Both functions are stateless (external cache threaded in/out; cross-attention K/V
recomputed each step; no cross-attention mask, since the full 30 s window means
every encoder frame is valid). Attention is spelled out manually — the Cohere
pipeline found `coreai_torch` mis-converts `F.scaled_dot_product_attention`.

## Self-contained asset

`convert.py` writes two files INSIDE the `.aimodel` dir so the runtime needs no
sidecars:

- **`tokens.txt`** — the byte-level BPE vocab, one token per line (line index ==
  id), in GPT-2 byte↔unicode form. The Swift `WhisperTokenizer` reverses that
  mapping to raw bytes and UTF-8 decodes.
- **`whisper.json`** — architecture constants (n_mels, mel_frames, T_enc, d, n_layer,
  n_head, vocab) + special token ids (sot/eot/transcribe/notimestamps + a
  language→id map). The engine reads all traced shapes from here, so a size swap
  (base → small → large-v3) needs no Swift change.

## Setup

Standalone, like the other convert dirs — `uv` + Python 3.12 + the pre-release
Apple `coreai` wheels:

```sh
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -e .                      # pulls the pre-release coreai wheels
uv pip install "pydantic>=2.6,<2.10"     # required for the --quantize path
```

Whisper is **not gated** — no `huggingface-cli login` needed.

## Convert

```sh
python -m whisper_convert.convert --model-id openai/whisper-base --output whisper-base.aimodel
python -m whisper_convert.convert --model-id openai/whisper-tiny --quantize palette4 --output whisper-tiny-palette4.aimodel
```

Then run it through the CLI (backend auto-detected from `whisper_decode`):

```sh
voixful transcribe clip.wav --model whisper-base.aimodel
```

`--quantize {none,palette4,palette6}` palettizes the encoder (the RAM/latency win;
see `../README.md`).

## License

- Conversion code → Apache-2.0 (repo root `LICENSE` / `NOTICE`).
- `.aimodel` weights → derived from the upstream Whisper checkpoint
  (`openai/whisper-*`, MIT); retain its license + attribution when redistributing.
