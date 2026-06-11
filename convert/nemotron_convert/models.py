"""PyTorch nn.Module wrappers exposing Nemotron streaming caches as register_buffer state. See PLAN.md §3 Step 2 and §4."""

from __future__ import annotations

import torch
from torch import nn


class NemotronEncoderForCoreAI(nn.Module):
    """Wraps NeMo's cache-aware FastConformer encoder so the four streaming caches are
    `register_buffer` state — the pattern `coreai_torch` lowers to runtime `MutableViews`.

    Cache shapes are constants of the published checkpoint per PLAN.md §2 (do not
    parameterize): `pre_cache` carries trailing mel frames into the next chunk,
    `cache_last_channel` is the per-layer self-attention K/V cache, `cache_last_time`
    is the per-layer convolution left-pad cache, and `cache_last_channel_len` tracks
    the running attention-cache length.
    """

    def __init__(
        self,
        nemo_encoder: nn.Module,
        d_model: int = 1024,
        n_layers: int = 24,
    ) -> None:
        super().__init__()
        self.inner = nemo_encoder
        self.d_model = d_model
        self.n_layers = n_layers

        # VERIFY: d_model 512 vs 1024 — see PLAN.md §2 footnote and §10 open question 2.
        # HF model card says 512 (likely projection / acoustic embedding); soniqo/speech-swift's
        # cache shape [24, 1, 56, 1024] implies encoder hidden = 1024. Confirm in Phase 1 by
        # inspecting the loaded NeMo cfg and update both the default here and PLAN.md §2.

        # Four streaming caches per PLAN.md §2 cache tensor shapes table.
        self.register_buffer("pre_cache", torch.zeros(1, 128, 9, dtype=torch.float32))
        self.register_buffer(
            "cache_last_channel",
            torch.zeros(n_layers, 1, 56, d_model, dtype=torch.float32),
        )
        self.register_buffer(
            "cache_last_time",
            torch.zeros(n_layers, 1, d_model, 8, dtype=torch.float32),
        )
        self.register_buffer(
            "cache_last_channel_len",
            torch.zeros(1, dtype=torch.int32),
        )

    def forward(
        self,
        mel_chunk: torch.Tensor,
        lang_id: torch.Tensor,
    ) -> torch.Tensor:
        # VERIFY: NeMo encoder API — confirm streaming_forward signature; if absent,
        # hand-write cache plumbing per PLAN.md §3 Step 2. The signature below is the
        # ideal shape we want; Phase 2 confirms by inspecting NeMo's encoder forward
        # (PLAN.md §10 open question 3).
        enc_out, new_pre, new_lc, new_lt, new_len = self.inner.streaming_forward(
            mel_chunk,
            lang_id,
            self.pre_cache,
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_len,
        )

        # In-place copy_ is the predefined pattern the coreai_torch converter recognizes
        # and lowers to a stateful runtime input. See PLAN.md §4 "Authoring side (PyTorch)".
        self.pre_cache.copy_(new_pre)
        self.cache_last_channel.copy_(new_lc)
        self.cache_last_time.copy_(new_lt)
        self.cache_last_channel_len.copy_(new_len)

        return enc_out


class NemotronPredictionForCoreAI(nn.Module):
    """Wraps the 2-layer LSTM prediction network so the hidden and cell states are
    `register_buffer`s. Shapes per PLAN.md §2 (pred_hidden = 640) and §4 "Prediction
    LSTM state".
    """

    def __init__(
        self,
        nemo_prediction: nn.Module,
        pred_hidden: int = 640,
        n_lstm_layers: int = 2,
    ) -> None:
        super().__init__()
        self.inner = nemo_prediction
        self.pred_hidden = pred_hidden
        self.n_lstm_layers = n_lstm_layers

        # LSTM hidden + cell state: [n_layers, batch=1, pred_hidden].
        self.register_buffer(
            "lstm_h",
            torch.zeros(n_lstm_layers, 1, pred_hidden, dtype=torch.float32),
        )
        self.register_buffer(
            "lstm_c",
            torch.zeros(n_lstm_layers, 1, pred_hidden, dtype=torch.float32),
        )

    def forward(self, prev_token: torch.Tensor) -> torch.Tensor:
        # VERIFY: NeMo decoder/prediction API — confirm the exact entry point and arg
        # order. NeMo's RNNT decoder typically takes (prev_token, states) and returns
        # (pred_emb, new_states); confirm names in Phase 2 (PLAN.md §10 open question 3).
        pred_emb, (new_h, new_c) = self.inner(
            prev_token,
            (self.lstm_h, self.lstm_c),
        )

        self.lstm_h.copy_(new_h)
        self.lstm_c.copy_(new_c)

        return pred_emb


class NemotronJoinerForCoreAI(nn.Module):
    """RNN-T joiner: pure function combining a single encoder step with a single
    prediction embedding into vocab logits. Vocab size 13,087 includes lang tags and
    the blank id (13087). See PLAN.md §2.
    """

    def __init__(self, nemo_joiner: nn.Module) -> None:
        super().__init__()
        self.inner = nemo_joiner

    def forward(
        self,
        enc_step: torch.Tensor,
        pred_emb: torch.Tensor,
    ) -> torch.Tensor:
        # VERIFY: NeMo joint API — confirm method name and arg order. NeMo's RNNT joint
        # is typically `joint(encoder_outputs, decoder_outputs)`; confirm in Phase 2.
        return self.inner(enc_step, pred_emb)
