"""torch.export entrypoints for the Qwen3-ASR wrappers (hidden read from config)."""

from __future__ import annotations

import torch
from torch.export import ExportedProgram

import coreai_torch

from .models import MAX_SEQ, Qwen3ASRAudioEncoderForCoreAI


def _decomp(ep: ExportedProgram) -> ExportedProgram:
    return ep.run_decompositions(coreai_torch.get_decomp_table())


def export_encoder(wrapper: Qwen3ASRAudioEncoderForCoreAI, mel_frames: int) -> ExportedProgram:
    mel = torch.zeros(1, wrapper.n_mels, mel_frames, dtype=torch.float32)
    length = torch.tensor([mel_frames], dtype=torch.int32)
    return _decomp(torch.export.export(wrapper, args=(mel, length)))


def export_embed(wrapper, n: int) -> ExportedProgram:
    ids = torch.zeros(1, n, dtype=torch.int32)
    return _decomp(torch.export.export(wrapper, args=(ids,)))


def export_prefill(wrapper, hidden: int) -> ExportedProgram:
    embeds = torch.zeros(1, wrapper.seq_len, hidden, dtype=torch.float32)
    last_idx = torch.tensor([wrapper.seq_len - 1], dtype=torch.int32)
    return _decomp(torch.export.export(wrapper, args=(embeds, last_idx)))


def export_step(wrapper) -> ExportedProgram:
    tok = torch.zeros(1, dtype=torch.int32)
    pos = torch.tensor([MAX_SEQ - 1], dtype=torch.int32)
    return _decomp(torch.export.export(wrapper, args=(tok, pos)))
