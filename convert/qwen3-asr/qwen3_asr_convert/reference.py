"""Load the Qwen3-ASR checkpoint via Qwen's own `qwen_asr` modeling code.

The published `Qwen/Qwen3-ASR-*` checkpoints use the `thinker.*` naming from
Qwen's `qwen-asr` package; released stock `transformers` renamed/reshaped the
class and cannot load them. Importing `qwen_asr.inference.qwen3_asr` registers
the correct config/model/processor with Auto*, after which AutoModel loads the
checkpoint cleanly.
"""

from __future__ import annotations

import torch

MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
EOS = 151645          # <|im_end|>
AUDIO_PAD = 151676    # <|audio_pad|> placeholder token


def load_model(model_id: str = MODEL_ID):
    import qwen_asr.inference.qwen3_asr  # noqa: F401 — registers qwen3_asr with Auto*
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(model_id, dtype=torch.float32).eval()
    processor = AutoProcessor.from_pretrained(model_id, fix_mistral_regex=True)
    return model, processor
