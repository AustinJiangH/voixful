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
    chunk_frames: int = 128,
) -> ExportedProgram:
    """Export the encoder wrapper at ONE fixed chunk length (`chunk_frames` mel
    frames). Dynamic-shape export hits a coreai_torch lowering bug on
    `aten.repeat` with a symbolic count, so the runtime instead selects among
    several fixed-shape window programs (`encode` = canonical 128-frame chunk;
    `encode_<frames>` variants). The cache-aware FastConformer accepts any
    chunk length against the SAME cache tensors — cache shapes derive from the
    left context, which doesn't change per window — so all windows share the
    streaming state. 32 frames = the 320 ms dictation/captioning cadence;
    256 = batch-throughput chunk (pairs with a future baked preprocessor).
    """
    example_mel = torch.zeros(1, 128, chunk_frames, dtype=torch.float32)
    example_length = torch.tensor([chunk_frames], dtype=torch.int32)

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
