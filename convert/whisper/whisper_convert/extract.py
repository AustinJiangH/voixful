"""Load an OpenAI Whisper checkpoint and hand back the eval-mode model.

Whisper ships as a standard `WhisperForConditionalGeneration` in HF transformers
(no `trust_remote_code`, not gated). `openai/whisper-base` is the smallest
reasonable default; `-tiny` is fine for a first bring-up and `-large-v3` is the
128-mel variant.
"""

from __future__ import annotations

import torch

DEFAULT_MODEL_ID = "openai/whisper-base"


def load_whisper(model_id: str = DEFAULT_MODEL_ID):
    """Return the eval-mode `WhisperForConditionalGeneration` in fp32 on CPU."""
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )
    model.eval()
    return model
