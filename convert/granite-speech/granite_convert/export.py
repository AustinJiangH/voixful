"""torch.export entrypoints for the Granite NAR wrappers."""

from __future__ import annotations

import torch
from torch.export import ExportedProgram

import coreai_torch

from .models import SAMPLES_PER_FRAME, GraniteEditForCoreAI, GraniteEncoderForCoreAI


def export_encoder(wrapper: GraniteEncoderForCoreAI) -> ExportedProgram:
    example_audio = torch.zeros(1, wrapper.frames * SAMPLES_PER_FRAME, dtype=torch.float32)
    exported = torch.export.export(wrapper, args=(example_audio,))
    return exported.run_decompositions(coreai_torch.get_decomp_table())


def export_edit(wrapper: GraniteEditForCoreAI) -> ExportedProgram:
    a, u = wrapper.audio_len, wrapper.text_len
    example_audio = torch.zeros(1, a, 2048, dtype=torch.float32)
    example_ids = torch.zeros(1, u, dtype=torch.int32)
    example_mask = torch.ones(1, a + u, dtype=torch.float32)
    exported = torch.export.export(wrapper, args=(example_audio, example_ids, example_mask))
    return exported.run_decompositions(coreai_torch.get_decomp_table())
