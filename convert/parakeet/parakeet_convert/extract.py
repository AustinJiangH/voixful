"""Pull submodules from a NeMo Parakeet TDT ASRModel for Core AI conversion.

Parakeet TDT 0.6B (`nvidia/parakeet-tdt-0.6b-v3` / `-v2`) is `EncDecRNNTBPEModel`
with a **full-context** `ConformerEncoder` (att_context_size [-1,-1] → no streaming
caches) and a **TDT** joint (token + duration logits, durations [0,1,2,3,4]).
"""

from __future__ import annotations

from typing import Any

import nemo.collections.asr as nemo_asr


def extract_modules_from_nemo(model_id: str) -> dict[str, Any]:
    model = nemo_asr.models.ASRModel.from_pretrained(model_id)

    # Full-context model: NO cache_support config (Parakeet TDT is offline/batch).
    # Disable the joint's fuse_loss_wer path so joint(f, g) returns logits directly.
    model.joint._fuse_loss_wer = False

    return {
        "encoder": model.encoder,
        "prediction": model.decoder,
        "joiner": model.joint,
        "preprocessor": model.preprocessor,
        "config": model.cfg,
        "tokenizer": model.tokenizer,
    }
