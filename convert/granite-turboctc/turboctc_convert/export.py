"""torch.export entrypoint for the TurboCTC wrapper."""

from __future__ import annotations

import torch
from torch.export import ExportedProgram

import coreai_torch

from .models import SAMPLES_PER_FRAME, TurboCTCForCoreAI, make_exportable


def export_ctc(wrapper: TurboCTCForCoreAI) -> ExportedProgram:
    make_exportable(wrapper.encoder)  # idempotent; survives palettizer copies
    samples = wrapper.frames * SAMPLES_PER_FRAME
    example_audio = torch.zeros(1, samples, dtype=torch.float32)
    example_length = torch.tensor([samples], dtype=torch.int32)
    exported = torch.export.export(wrapper, args=(example_audio, example_length))
    return exported.run_decompositions(coreai_torch.get_decomp_table())
