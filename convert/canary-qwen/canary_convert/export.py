"""torch.export entrypoints for the Canary-Qwen wrappers."""

from __future__ import annotations

import torch
from torch.export import ExportedProgram

import coreai_torch

from .models import (
    MAX_SEQ,
    CanaryEncoderForCoreAI,
    QwenEmbedForCoreAI,
    QwenPrefillForCoreAI,
    QwenStepForCoreAI,
)


def _decomp(ep: ExportedProgram) -> ExportedProgram:
    return ep.run_decompositions(coreai_torch.get_decomp_table())


def export_encoder(wrapper: CanaryEncoderForCoreAI, mel_frames: int) -> ExportedProgram:
    mel = torch.zeros(1, 128, mel_frames, dtype=torch.float32)
    length = torch.tensor([mel_frames], dtype=torch.int32)
    return _decomp(torch.export.export(wrapper, args=(mel, length)))


def export_embed(wrapper: QwenEmbedForCoreAI, n: int) -> ExportedProgram:
    ids = torch.zeros(1, n, dtype=torch.int32)
    return _decomp(torch.export.export(wrapper, args=(ids,)))


def _cache_example() -> tuple[torch.Tensor, torch.Tensor]:
    shape = (28, 1, 8, MAX_SEQ, 128)
    return torch.zeros(shape), torch.zeros(shape)


def export_prefill(wrapper: QwenPrefillForCoreAI) -> ExportedProgram:
    embeds = torch.zeros(1, wrapper.seq_len, 2048, dtype=torch.float32)
    last_idx = torch.tensor([wrapper.seq_len - 1], dtype=torch.int32)
    k, v = _cache_example()
    return _decomp(torch.export.export(wrapper, args=(embeds, last_idx, k, v)))


def export_step(wrapper: QwenStepForCoreAI) -> ExportedProgram:
    tok = torch.zeros(1, dtype=torch.int32)
    pos = torch.tensor([MAX_SEQ - 1], dtype=torch.int32)
    k, v = _cache_example()
    return _decomp(torch.export.export(wrapper, args=(tok, pos, k, v)))
