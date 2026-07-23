"""Pulls submodules from a NeMo ASRModel for Core AI conversion. VERIFY: inspect model.cfg to resolve d_model=512 vs 1024 (open question)."""

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
      - preprocessor: NeMo's AudioToMelSpectrogramPreprocessor
      - config: the model's `cfg` (OmegaConf DictConfig) — inspect to resolve
        the d_model 512 vs 1024 ambiguity and the preprocessor settings
      - tokenizer: SentencePiece tokenizer (vocab 13,087)
    """

    model = nemo_asr.models.ASRModel.from_pretrained(model_id)

    # Enable cache support BEFORE pulling out the encoder. This is the toggle that
    # makes forward_for_export accept and return the streaming caches. Without it,
    # the exported graph would silently drop streaming state (the NeMo
    # "set_export_config" pattern — verified 2026-06-11
    # against the english-only sibling).
    model.set_export_config({"cache_support": "True"})

    # Disable the joint's fuse_loss_wer path so `joint(f, g)` returns logits
    # directly rather than requiring transcript+length kwargs (a training-time
    # convenience that gets in the way of inference export).
    model.joint._fuse_loss_wer = False

    # VERIFY: NeMo attribute names — this package uses .encoder / .decoder / .joint
    # / .preprocessor / .cfg / .tokenizer. Confirm against the actual loaded class;
    # some NeMo model classes expose `prediction_network` or `joint_net` aliases.

    # VERIFY: mel preprocessor exportability (open question) — whether
    # `torch.export` accepts NeMo's AudioToMelSpectrogramPreprocessor (the `make_seq_mask_like`
    # op blocked coremltools). This package returns the preprocessor for
    # callers to test exportability (Branch A); if Branch A fails, the
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
