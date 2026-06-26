"""Load Cohere Transcribe and hand back the eval-mode model for wrapping.

The checkpoint ships custom modeling code (`modeling_cohere_asr.py`), so it loads
via `trust_remote_code=True`. The repo is gated — authenticate first
(`hf auth login` / `HF_TOKEN`).
"""

from __future__ import annotations

import torch

DEFAULT_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"


def load_cohere(model_id: str = DEFAULT_MODEL_ID):
    """Return the eval-mode `CohereAsrForConditionalGeneration` in fp32 on CPU."""
    from transformers import AutoModelForSpeechSeq2Seq

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    return model
