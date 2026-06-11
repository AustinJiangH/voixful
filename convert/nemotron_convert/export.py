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

    mel_T = torch.export.Dim("mel_T", min=1, max=max_chunk_frames)

    # Example inputs. Shapes per PLAN.md §2:
    # - mel_chunk: [batch=1, mel_bins=128, T]
    # - lang_id: 128-dim one-hot language ID (PLAN.md §2 "language-prompt fusion")
    # VERIFY: lang_id encoding — confirm whether NeMo wants the one-hot vector, an
    # integer index, or a string; PLAN.md §6 has it as a 128-dim one-hot but the
    # exact dtype/shape is not nailed down. Inspect the language-prompt fusion code
    # path in Phase 1/2.
    example_mel = torch.zeros(1, 128, max_chunk_frames, dtype=torch.float32)
    example_lang = torch.zeros(1, 128, dtype=torch.float32)

    exported = torch.export.export(
        wrapper,
        args=(example_mel, example_lang),
        dynamic_shapes={"mel_chunk": {2: mel_T}, "lang_id": None},
    )
    return exported.run_decompositions(coreai_torch.get_decomp_table())


def export_prediction(wrapper: NemotronPredictionForCoreAI) -> ExportedProgram:
    """Export the prediction wrapper. The previous token is a scalar int32; the LSTM
    state is carried via `register_buffer` (see `models.py`).
    """

    # Single previous token, batch=1. int32 matches the cache_last_channel_len dtype
    # convention used elsewhere in PLAN.md §2.
    # VERIFY: prev_token shape — NeMo's RNNT prediction typically takes a [B] or [B,1]
    # integer tensor. Confirm in Phase 2 against the actual decoder API.
    example_prev_token = torch.zeros(1, dtype=torch.int32)

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
    """Export the joiner. Pure function: takes one encoder step and one prediction
    embedding, returns vocab logits.
    """

    # VERIFY: encoder step shape — PLAN.md §2 implies enc_out is [B, T, d_model] from
    # the encoder; a single time step is [B, d_model] or [B, 1, d_model]. Confirm in
    # Phase 2 against NeMo's joint signature.
    # VERIFY: pred_emb shape — NeMo's prediction typically emits [B, 1, pred_hidden].
    example_enc_step = torch.zeros(1, 1, d_model, dtype=torch.float32)
    example_pred_emb = torch.zeros(1, 1, pred_hidden, dtype=torch.float32)

    exported = torch.export.export(
        wrapper,
        args=(example_enc_step, example_pred_emb),
    )
    return exported.run_decompositions(coreai_torch.get_decomp_table())
