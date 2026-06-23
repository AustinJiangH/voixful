"""torch.export each of the three wrapper modules with dynamic shapes and the coreai_torch decomposition table. See PLAN.md §3 Step 3."""

from __future__ import annotations

import coreai_torch
import torch
from torch.export import ExportedProgram

from .models import (
    NemotronEncoderForCoreAI,
    NemotronJoinerForCoreAI,
    NemotronPredictionForCoreAI,
)


def export_encoder(
    wrapper: NemotronEncoderForCoreAI,
    max_chunk_frames: int = 112,
) -> ExportedProgram:
    """Export the encoder wrapper with a dynamic time axis covering all five chunk
    sizes from PLAN.md §2 (worst case 1120 ms = 112 frames at 10 ms hop).

    VERIFY: dynamic-shape survival — PLAN.md §10 open question 5 asks whether the
    dynamic `mel_T` dimension survives `coreai_torch` conversion. If conversion
    rejects it, fall back to one exported program per chunk size (5 total) per
    PLAN.md §3 Step 3 and Phase 2 final task.
    """

    # Fixed-shape export: T=128 (the smallest multiple of 8 ≥ 121, the streaming-cfg
    # minimum). Dynamic-shape export hits a coreai_torch 0.4.0 lowering bug on
    # `aten.repeat` whose count is a symbolic int rather than a Python int — the
    # framework's `replace_repeat` decomposition assumes a literal. Until that's
    # fixed upstream, ship one fixed-shape `.aimodel` per chunk size that matters.
    # The 128-frame chunk corresponds to the larger (1120 ms-ish) streaming
    # variant; smaller chunks would re-export with `example_T=8*k` for k∈{15..16}.
    example_T = 128
    example_mel = torch.zeros(1, 128, example_T, dtype=torch.float32)
    example_length = torch.tensor([example_T], dtype=torch.int32)

    exported = torch.export.export(
        wrapper,
        args=(example_mel, example_length),
    )
    return exported.run_decompositions(coreai_torch.get_decomp_table())


def export_prediction(wrapper: NemotronPredictionForCoreAI) -> ExportedProgram:
    """Export the prediction wrapper. `decoder.predict` accepts a [B, U=1] int32
    token tensor; the LSTM state is carried via `register_buffer`.
    """

    # Single previous token shaped [B=1, U=1] int32.
    example_prev_token = torch.zeros(1, 1, dtype=torch.int32)

    exported = torch.export.export(
        wrapper,
        args=(example_prev_token,),
    )
    return exported.run_decompositions(coreai_torch.get_decomp_table())


def export_joiner(
    wrapper: NemotronJoinerForCoreAI,
    d_model: int = 1024,
    pred_hidden: int = 640,
) -> ExportedProgram:
    """Export the joiner. `joint(f, g)` takes channels-last tensors:
    - enc_step: `[B=1, T=1, d_model=1024]`
    - pred_emb: `[B=1, U=1, pred_hidden=640]`
    Output: `[B, T=1, U=1, vocab=1025]` (vocab 1024 SP tokens + 1 blank).
    """

    example_enc_step = torch.zeros(1, 1, d_model, dtype=torch.float32)
    example_pred_emb = torch.zeros(1, 1, pred_hidden, dtype=torch.float32)

    exported = torch.export.export(
        wrapper,
        args=(example_enc_step, example_pred_emb),
    )
    return exported.run_decompositions(coreai_torch.get_decomp_table())
