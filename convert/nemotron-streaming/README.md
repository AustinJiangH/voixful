# convert/ — NeMo -> torch.export -> coreai_torch

Python pipeline that turns `nvidia/nemotron-speech-streaming-en-0.6b` into a single
multi-function `.aimodel` containing three named inference functions: `encode`,
`predict_step`, and `join`. The `VERIFY` comments throughout this package track the
remaining open questions.

## Install

**Recommended (verified 2026-06-11):** use `uv` + Python 3.12. `uv` handles the
pre-release pin (`coreai-torch` depends on `coreai-core==1.0.0b1`, which plain
`pip install` skips by default).

```sh
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -e .
uv pip install "pydantic>=2.6,<2.10"   # required for the --quantize path; see ../README.md
```

**Alternative (pip):** plain pip works with `--pre`:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install --pre -e .
```

NeMo 26.06 and `coreai-torch` (**≥ 0.4.1** — 0.4.0 exports fail to load from
macOS 27 beta 3 on) both run cleanly on the same PyTorch 2.11.0 / Python 3.12
install — no two-env workaround required. Python 3.13/3.14 wheels for several
NeMo deps are spotty as of 2026-06; stick to 3.12.

## Auth

The Nemotron checkpoint may be gated. If `nemo_asr.models.ASRModel.from_pretrained`
fails to download weights, log in to Hugging Face first:

```sh
huggingface-cli login
```

## Run

```sh
python -m nemotron_convert.convert --output ./nemotron-streaming.aimodel
```

Optional flags:

- `--model-id` (default `nvidia/nemotron-speech-streaming-en-0.6b`)
- `--max-chunk-frames` (default `112`, i.e., 1120 ms at a 10 ms hop)

Output: a single FP16 `.aimodel` at `--output`. Add `--quantize palette4` for the
4-bit palettized model (7× smaller disk + RAM, faster) — see `../README.md`.

## License

- This **conversion code** → Apache License 2.0 (repo root `LICENSE` / `NOTICE`).
- The **`.aimodel` weights** → derived from `nvidia/nemotron-speech-streaming-en-0.6b`,
  governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
  See `../../MODEL_CARD.md`.

## Verify parity

```sh
python -m nemotron_convert.verify_parity \
    --aimodel ./nemotron-streaming.aimodel \
    --wav ./fixtures/sample.wav
```

Checks: encoder SNR >= 40 dB vs. the NeMo reference and exact token-sequence match
on the same 30-second clip.
