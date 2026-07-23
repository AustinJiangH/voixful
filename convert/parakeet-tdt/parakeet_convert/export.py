"""torch.export the three Parakeet wrappers with the coreai_torch decomposition table."""

from __future__ import annotations

import coreai_torch
import torch
from torch.export import ExportedProgram

from .models import (
    ParakeetEncoderForCoreAI,
    ParakeetJoinerForCoreAI,
    ParakeetPredictionForCoreAI,
)

# Parakeet is full-context (not chunked), so the encoder takes the WHOLE clip's
# mel frames at once. coreai_torch 0.4.0 can't carry a dynamic time axis (the
# `aten.repeat` symbolic-int lowering bug), so we export a fixed mel length and
# pad/truncate audio to it at runtime. 1000 frames ≈ 10 s at a 10 ms hop.
DEFAULT_MEL_FRAMES = 1000

# TDT joint output width = vocab + 1 blank + 5 durations (v3: 8198, v2: 1030).


def export_encoder(
    wrapper: ParakeetEncoderForCoreAI, mel_frames: int = DEFAULT_MEL_FRAMES
) -> ExportedProgram:
    example_mel = torch.zeros(1, 128, mel_frames, dtype=torch.float32)
    example_length = torch.tensor([mel_frames], dtype=torch.int32)
    exported = torch.export.export(wrapper, args=(example_mel, example_length))
    return exported.run_decompositions(coreai_torch.get_decomp_table())


def export_prediction(wrapper: ParakeetPredictionForCoreAI) -> ExportedProgram:
    example_prev_token = torch.zeros(1, 1, dtype=torch.int32)
    exported = torch.export.export(wrapper, args=(example_prev_token,))
    return exported.run_decompositions(coreai_torch.get_decomp_table())


def export_joiner(
    wrapper: ParakeetJoinerForCoreAI, d_model: int = 1024, pred_hidden: int = 640
) -> ExportedProgram:
    example_enc_step = torch.zeros(1, 1, d_model, dtype=torch.float32)
    example_pred_emb = torch.zeros(1, 1, pred_hidden, dtype=torch.float32)
    exported = torch.export.export(wrapper, args=(example_enc_step, example_pred_emb))
    return exported.run_decompositions(coreai_torch.get_decomp_table())


def export_enc_projector(wrapper, t_out: int = 125, d_model: int = 1024) -> ExportedProgram:
    example_enc = torch.zeros(1, t_out, d_model, dtype=torch.float32)
    exported = torch.export.export(wrapper, args=(example_enc,))
    return exported.run_decompositions(coreai_torch.get_decomp_table())


def export_join_fused(wrapper, joint_hidden: int = 640, pred_hidden: int = 640) -> ExportedProgram:
    example_enc_proj_t = torch.zeros(1, 1, joint_hidden, dtype=torch.float32)
    example_pred_emb = torch.zeros(1, 1, pred_hidden, dtype=torch.float32)
    exported = torch.export.export(wrapper, args=(example_enc_proj_t, example_pred_emb))
    return exported.run_decompositions(coreai_torch.get_decomp_table())
