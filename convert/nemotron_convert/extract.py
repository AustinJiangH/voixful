"""Pulls submodules from a NeMo ASRModel for Core AI conversion. VERIFY: inspect model.cfg to resolve d_model=512 vs 1024 per PLAN.md §10 open question 2."""

from __future__ import annotations

from typing import Any

import nemo.collections.asr as nemo_asr


def extract_modules_from_nemo(model_id: str) -> dict[str, Any]:
    """Download (if needed) and load the Nemotron checkpoint, returning the raw
    submodules Core AI conversion needs.

    Returns a dict with keys:
      - encoder: the cache-aware FastConformer (`nn.Module`)
      - prediction: the 2-layer LSTM prediction network (`nn.Module`)
      - joiner: the RNN-T joint (`nn.Module`)
      - preprocessor: NeMo's AudioToMelSpectrogramPreprocessor (see PLAN.md §5)
      - config: the model's `cfg` (OmegaConf DictConfig) — inspect to resolve
        the d_model 512 vs 1024 ambiguity per PLAN.md §10 open question 2 and the
        preprocessor settings per §5
      - tokenizer: SentencePiece tokenizer (vocab 13,087)
    """

    model = nemo_asr.models.ASRModel.from_pretrained(model_id)

    # VERIFY: NeMo attribute names — PLAN.md §3 Step 1 uses .encoder / .decoder / .joint
    # / .preprocessor / .cfg / .tokenizer. Confirm against the actual loaded class in
    # Phase 1; some NeMo model classes expose `prediction_network` or `joint_net` aliases.

    # VERIFY: mel preprocessor exportability — PLAN.md §10 open question 4 asks whether
    # `torch.export` accepts NeMo's AudioToMelSpectrogramPreprocessor (the `make_seq_mask_like`
    # op blocked coremltools per PLAN.md §5). This package returns the preprocessor for
    # callers to test exportability in Phase 4 (Branch A); if Branch A fails, the
    # preprocessor is reimplemented via TorchMetalKernel (Branch B) or in Swift+Accelerate
    # (Branch C), and convert.py does not pack it into the .aimodel.
    return {
        "encoder": model.encoder,
        "prediction": model.decoder,
        "joiner": model.joint,
        "preprocessor": model.preprocessor,
        "config": model.cfg,
        "tokenizer": model.tokenizer,
    }
