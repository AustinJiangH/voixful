# convert/ — NeMo -> torch.export -> coreai_torch

Python pipeline that turns `nvidia/nemotron-3.5-asr-streaming-0.6b` into a single
multi-function `.aimodel` containing three named inference functions: `encode`,
`predict_step`, and `join`. See `../PLAN.md` §3 for the full design rationale and §10
for the open questions that the `VERIFY` comments throughout this package track.

## Install

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**PyTorch version conflict (PLAN.md §10 open question 1).** NeMo 26.06 pins a
specific PyTorch release, while `coreai_torch` requires the `torch.export` API
introduced in PyTorch 2.5+. If `pip install -e .` fails to resolve a single PyTorch
version that satisfies both, fall back to a two-environment workaround: load the
checkpoint in a NeMo-compatible env (env A), serialize the encoder / prediction /
joiner state dicts plus the model config to disk, then reload the modules in a
PyTorch-2.5+/coreai_torch env (env B) and run `torch.export` + conversion there.
Phase 1 of the plan resolves which environment shape is required.

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
