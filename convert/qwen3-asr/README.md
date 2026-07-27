# convert/qwen3-asr — Qwen3-ASR 0.6B/1.7B → Core AI `.aimodel`

Converts **Alibaba/Qwen Qwen3-ASR** (`Qwen/Qwen3-ASR-0.6B`, `Qwen/Qwen3-ASR-1.7B`)
into a Core AI `.aimodel`. This is a second **SALM** (Speech-Augmented Language
Model) backend alongside Canary-Qwen — an audio encoder feeding a **Qwen3** LLM
decoder — but multilingual (52 languages/dialects, with built-in language ID) and
available in two small sizes. Pure ASR: unlike Canary there is **no** second
formatting pass.

> **Status: scaffold.** This directory pins the conversion *contract* (deps, entry
> points, asset layout) and the Swift runtime (`VoixfulQwen3ASR`) is fully wired to
> it, but `convert.py` is not yet a finished, verified exporter — no converted
> Qwen3-ASR `.aimodel` has been produced or published. Until one exists, the two
> `models.json` entries are `excludeFromDownloads: true` (import a locally-converted
> asset to run them, exactly as with Cohere). The scoring/size numbers in
> `models.json` for the 0.6B variant are provisional estimates.

## Why this model

- **Multilingual small-tier** — fills Voixful's gap between English-only small
  models (Nemotron, Parakeet v2, Canary) and the heavier 2B+ multilingual models
  (Cohere). Qwen3-ASR-1.7B is competitive with top commercial APIs; the 0.6B
  variant is the family's best accuracy/efficiency trade-off.
- **SALM architecture** — reuses the same audio-encoder → Qwen3-decoder →
  KV-cache prefill/step shape the Canary backend already exercises.

## `.aimodel` entrypoints (the detection + engine contract)

The Swift engine (`Sources/VoixfulEngine/Qwen3ASR/Qwen3ASREngine.swift`) and the
backend detector (`step_qwen3` is the classification key) expect:

```
encode_<T>(mel [1,128,T] + length)  -> audio_embeds [1, T/8, H]
embed_<N>(token ids [1,N])          -> embeds                 # prompt/transcript splice
prefill_asr_<L>(embeds + last_idx)  -> first token            # writes K/V cache states
step_qwen3(one AR step)                                       # Qwen3 decode (transcription)
```

The `step_qwen3` name is deliberately distinct from Canary's `step_asr` so the two
SALM backends are told apart by function signature alone. Bundle `qwen3_asr.json`
(prompt template ids, eos, window + bucket tables) and `tokens.json` (id → base64
bytes for host detokenize) **inside** the `.aimodel` dir so the asset is
self-contained.

## Setup

Standalone, like the other convert dirs — **do not share a venv**.

```sh
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -e .                      # transformers + the pre-release coreai wheels
uv pip install --pre coreai-opt          # required for the --quantize path
uv pip install "pydantic>=2.6,<2.10"
```

## Convert

```sh
python -m qwen3_asr_convert.convert --model Qwen/Qwen3-ASR-0.6B \
    --output qwen3-asr-0.6b-palette4.aimodel \
    [--windows 1000,2000,4000] [--quantize none|palette4|palette6] \
    [--check-wav path.wav]
```

## Compute placement & accuracy (measured, M-series)

The converted asset is numerically correct — on-device it transcribes the fixture
**exactly** on the **CPU** compute unit. On **ANE/GPU** the fp16 compute flips
argmax on ambiguous tokens (measured: *"a test of the speech"* → *"the death of
the"*). This is an fp16-**compute** effect, not weight quantization: CPU runs the
same palette weights and is exact.

Isolation done during bring-up: keeping the tied `embed_tokens`/`lm_head`/`norm`
at full precision (palettizing only the 28 decoder layers) did **not** change the
ANE result — so the fp16 sensitivity is upstream, in the **audio encoder**, not
the logit head: feeding CPU-computed audio embeds to the ANE decoder transcribes
the fixture exactly. (The head-fp16 variant only inflated the asset 532 MB →
1.0 GB for no accuracy gain, so it was reverted.)

**The Swift engine implements the split** (`Qwen3ASREngine` specializes the asset
twice — encoder instance + decoder instance): the **default is encode→CPU,
decode→ANE**, which is both correct and the fastest correct placement. Override
per instance with `VOIXFUL_COMPUTE` (encoder) and `VOIXFUL_COMPUTE_DECODE`
(decoder); if they resolve to the same unit the two instances coalesce.

Timings (release, 5.5 s clip): mel 3 ms + encode **CPU 0.38 s** + decode **ANE
2.36 s** ≈ **2.75 s** and correct. (encode ANE ≈ 1.3 s but wrong; decode CPU ≈
3.9 s, decode GPU ≈ 2.37 s — both correct.) The remaining decode cost is the
per-token AR dispatch floor; a `step_qwen3_x4`-style unroll (Cohere's `decode_x4`)
is the next speed lever and would need matching exporter + engine work.

## License

- Conversion code → Apache-2.0 (repo root `LICENSE` / `NOTICE`).
- `.aimodel` weights → derived from `Qwen/Qwen3-ASR-*` (**Apache-2.0** upstream);
  retain the upstream license + attribution when redistributing.
