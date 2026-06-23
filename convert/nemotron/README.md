# convert/ — NeMo -> torch.export -> coreai_torch

Python pipeline that turns `nvidia/nemotron-3.5-asr-streaming-0.6b` into a single
multi-function `.aimodel` containing three named inference functions: `encode`,
`predict_step`, and `join`. See `../PLAN.md` §3 for the full design rationale and §10
for the open questions that the `VERIFY` comments throughout this package track.

## Install

**Recommended (verified 2026-06-11):** use `uv` + Python 3.12. `uv` handles the
pre-release pin (`coreai-torch` depends on `coreai-core==1.0.0b1`, which plain
`pip install` skips by default).

```sh
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

**Alternative (pip):** plain pip works with `--pre`:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install --pre -e .
```

Resolves the PLAN.md §10 open question 1: NeMo 26.06 and `coreai-torch` 0.4.0
both run cleanly on the same PyTorch 2.11.0 / Python 3.12 install — no two-env
workaround required. Python 3.13/3.14 wheels for several NeMo deps are spotty
as of 2026-06; stick to 3.12.

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

- `--model-id` (default `nvidia/nemotron-3.5-asr-streaming-0.6b`)
- `--max-chunk-frames` (default `112`, i.e., 1120 ms at a 10 ms hop)

Output: a single FP16 `.aimodel` at `--output`. No `coreai_opt` is invoked; PLAN.md
§3 Step 5b documents the conditional quantization path if Phase 6 measures
RTF > 0.3 on M2 or iOS deployment lands in scope.

## Verify parity

```sh
python -m nemotron_convert.verify_parity \
    --aimodel ./nemotron-streaming.aimodel \
    --wav ./fixtures/sample.wav
```

Checks: encoder SNR >= 40 dB vs. the NeMo reference and exact token-sequence match
on the same 30-second clip. See PLAN.md §3 Step 6 for the rationale and tolerance.
