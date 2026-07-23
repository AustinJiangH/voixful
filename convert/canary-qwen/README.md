# convert/canary-qwen — Canary-Qwen 2.5B → Core AI `.aimodel`

Converts **NVIDIA Canary-Qwen 2.5B** (`nvidia/canary-qwen-2.5b`) into a Core AI
`.aimodel`. This is Voixful's **SALM** (Speech-Augmented Language Model) backend —
a Conformer encoder feeding a **Qwen3-1.7B** LLM decoder — the largest and most
accurate-at-punctuation model we ship. Prefer it for file transcripts (its decode
is state-transfer-bound, so live-mic passes are sluggish).

## Why this model

- **Best punctuation + capitalization** of the set — the Qwen3 backbone produces
  fully formatted transcripts, and an optional second LLM pass reformats them as
  markdown bullets (`voixful transcribe --bullets`).
- **SALM architecture** — exercises a real LLM decoder with a LoRA adapter
  (merged for ASR mode, dropped for the pristine-Qwen formatting pass) and full
  KV-cache prefill/step threading.

## `.aimodel` entrypoints

```
encode_<T>(mel [1,128,T] + length)  -> audio_embeds [1, T/8, 2048]
embed_256(token ids [1,256])        -> embeds                 # prompt/transcript splice
prefill_asr_<L>(embeds + last_idx)  -> first token            # writes K/V cache states
step_asr(one AR step)                                          # LoRA-merged Qwen3 (transcription)
prefill_llm_<L> / step_llm                                     # pristine Qwen3-1.7B-instruct (formatting)
```

The asset bundles `canary.json` (prompt template ids, eos, window and bucket
tables) and `tokens.json` (id → base64 bytes for host detokenize) inside the
`.aimodel` dir. `VoixfulCanaryQwen`'s `CanaryTranscriber` reads both.

## Setup

Standalone, like the other convert dirs. SALM lives in
`nemo.collections.speechlm2`, so this needs **NeMo ≥ 2.5** — **do not share a
venv** with the other converters.

```sh
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -e .                      # pulls NeMo + the pre-release coreai wheels
uv pip install --pre coreai-opt          # required for the --quantize path
uv pip install "pydantic>=2.6,<2.10"
```

## Convert

```sh
python -m canary_convert.convert --output canary-qwen-palette4.aimodel \
    [--windows 1000,2000,4000] [--quantize none|palette4|palette6] \
    [--check-wav path.wav]
```

Weight sharing: the encoder is palettized once; each Qwen3 variant (ASR-LoRA and
pristine) is palettized once as the bare HF model, and every prefill/step/embed
wrapper is built around the **same** palettized instance. A post-palettize
transcript self-check runs in torch **before** the slow export.

## License

- Conversion code → Apache-2.0 (repo root `LICENSE` / `NOTICE`).
- `.aimodel` weights → derived from `nvidia/canary-qwen-2.5b` (**CC-BY-4.0**
  upstream); retain the upstream license + attribution when redistributing.
