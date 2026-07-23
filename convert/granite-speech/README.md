# convert/granite-speech — Granite Speech 4.1 2B NAR → Core AI `.aimodel`

Converts **IBM Granite Speech 4.1 2B NAR** (`ibm-granite/granite-speech-4.1-2b-nar`)
into a Core AI `.aimodel`. This is Voixful's **non-autoregressive (NAR)** backend —
a CTC acoustic pass followed by a **single bidirectional LLM edit pass**, so there
is no decode loop at all (two dispatches per clip).

## Why this model

- **Best WER we ship** (4.95 Open ASR Leaderboard avg) while staying fast live,
  because the LLM refines the CTC hypothesis in one pass instead of decoding
  token-by-token.
- **Different runtime shape** from every other backend — no autoregressive greedy
  loop, no KV cache threading. Exercises the "acoustic model + LLM editor" pattern.
- Multilingual: **en, fr, de, es, pt**.

## `.aimodel` entrypoints

```
encode_<F>(audio window of F encoder frames, F*320 samples @ 16 kHz)
    -> (audio_embeds [1, F/5, 2048], ctc_ids [1, F/4] i32)

edit_<A>_<U>(one bidirectional LLM pass)
    -> text ids [1, U] i32          # A = F/5 (pairs with its window), U = text bucket
```

The asset bundles `granite.json` (blank id, min edit length, window/bucket table)
and `tokens.json` (id → base64 raw-byte table for host-side BPE detokenize) inside
the `.aimodel` dir, so the specialize cache keys on the final contents.
`VoixfulGraniteNAR`'s `GraniteTranscriber` reads both — the asset is self-contained.

## Setup

Standalone, like the other convert dirs. Granite Speech NAR ships
`trust_remote_code` modeling that needs **transformers v5** APIs, so **do not share
a venv** with `convert/cohere-transcribe` (which pins `<5`).

```sh
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -e .                      # pulls the pre-release coreai wheels
uv pip install --pre coreai-opt          # required for the --quantize path
uv pip install "pydantic>=2.6,<2.10"
```

## Convert

```sh
python -m granite_convert.convert --output granite-nar-palette4.aimodel \
    [--windows 200,400,800,1600] [--quantize none|palette4|palette6] \
    [--check-wav path.wav]
```

Weight sharing mirrors the other converters: the encoder/projector and the LLM are
palettized **once** (on the largest-window wrapper) and grafted into every other
window's wrapper, so the multi-window asset carries one copy of each weight set.
After palettization, a full-pipeline transcript self-check runs in pure torch
**before** the (slow) export and aborts on quality collapse.

## License

- Conversion code → Apache-2.0 (repo root `LICENSE` / `NOTICE`).
- `.aimodel` weights → derived from `ibm-granite/granite-speech-4.1-2b-nar`
  (**Apache-2.0** upstream); retain the upstream license + attribution when
  redistributing.
