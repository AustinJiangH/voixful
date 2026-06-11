"""PyTorch nn.Module wrappers exposing Nemotron streaming caches as register_buffer state.

Verified 2026-06-11 against `nvidia/nemotron-speech-streaming-en-0.6b` (the
English-only sibling). The 40-language `nvidia/nemotron-3.5-asr-streaming-0.6b`
checkpoint requires NeMo's unreleased `EncDecRNNTBPEModelWithPrompt` class.

Real cache + tensor shapes (from probing the loaded model):
  encoder.forward_for_export inputs:
    audio_signal:            [1, 128, T]              float32  (T must be multiple of 8)
    length:                  [1]                      int32
    cache_last_channel:      [1, 24, 70, 1024]        float32  (batch, layers, K/V, d_model)
    cache_last_time:         [1, 24, 1024, 8]         float32
    cache_last_channel_len:  [1]                      int64
  encoder.forward_for_export outputs:
    enc_out:                 [1, 1024, T_out]         float32  (channels-second!)
    enc_len:                 [1]                      int64
    + the three updated caches (same shapes as inputs)

  prediction (decoder.predict):
    inputs:  y=[B, U=1] int32, state=[h, c] where h/c are [2, B, 640] float32
    outputs: pred_emb [B, U=1, 640], new_state=[h, c]

  joint (.joint method, non-typed direct):
    inputs:  f=[B, 1024, T=1] (transposed enc), g=[B, 640, U=1] (transposed pred)
    output:  logits [B, T=1, U=1, vocab=1025]   (vocab 1024 + 1 blank)
"""

from __future__ import annotations

import torch
from torch import nn


class NemotronEncoderForCoreAI(nn.Module):
    """Wraps NeMo's cache-aware FastConformer encoder so the three streaming caches
    are `register_buffer` state. Uses `forward_for_export` which expects the
    batch-first cache layout (matching `input_example()` shapes).
    """

    def __init__(
        self,
        nemo_encoder: nn.Module,
        d_model: int = 1024,
        n_layers: int = 24,
        cache_channel_size: int = 70,
        cache_time_size: int = 8,
    ) -> None:
        super().__init__()
        self.inner = nemo_encoder
        self.d_model = d_model
        self.n_layers = n_layers

        self.register_buffer(
            "cache_last_channel",
            torch.zeros(1, n_layers, cache_channel_size, d_model, dtype=torch.float32),
        )
        self.register_buffer(
            "cache_last_time",
            torch.zeros(1, n_layers, d_model, cache_time_size, dtype=torch.float32),
        )
        self.register_buffer(
            "cache_last_channel_len",
            torch.zeros(1, dtype=torch.int64),
        )

    def forward(
        self,
        mel_chunk: torch.Tensor,
        mel_length: torch.Tensor,
    ) -> torch.Tensor:
        enc_out, _enc_len, new_lc, new_lt, new_len = self.inner.forward_for_export(
            mel_chunk,
            mel_length,
            self.cache_last_channel,
            self.cache_last_time,
            self.cache_last_channel_len,
        )
        self.cache_last_channel.copy_(new_lc)
        self.cache_last_time.copy_(new_lt)
        self.cache_last_channel_len.copy_(new_len)
        # `forward_for_export` returns channels-second [B, d_model, T_out]. Transpose
        # to channels-last [B, T_out, d_model] so the Swift greedy decoder can iterate
        # `t` over `shape[1]` (the convention assumed by GreedyRNNTDecoder.swift).
        return enc_out.transpose(1, 2).contiguous()


class NemotronPredictionForCoreAI(nn.Module):
    """Single-step RNN-T prediction. Wraps `decoder.predict(y, state, add_sos=False,
    batch_size=1)` which is NeMo's non-typed (positional-args-ok) entry point.
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

        self.register_buffer(
            "lstm_h",
            torch.zeros(n_lstm_layers, 1, pred_hidden, dtype=torch.float32),
        )
        self.register_buffer(
            "lstm_c",
            torch.zeros(n_lstm_layers, 1, pred_hidden, dtype=torch.float32),
        )

    def forward(self, prev_token: torch.Tensor) -> torch.Tensor:
        # Caller passes either [1] or [1, 1] int32. Normalize to [1, 1].
        if prev_token.dim() == 1:
            prev_token = prev_token.unsqueeze(0)

        # predict() accepts positional args. Output: (pred_emb [B, U=1, pred_hidden],
        # new_state=[h, c]).
        pred_emb, new_state = self.inner.predict(
            prev_token,
            [self.lstm_h, self.lstm_c],
            False,  # add_sos
            1,      # batch_size
        )

        self.lstm_h.copy_(new_state[0])
        self.lstm_c.copy_(new_state[1])

        return pred_emb


class NemotronJoinerForCoreAI(nn.Module):
    """RNN-T joiner. Uses `joint.joint(f, g)` — the non-typed direct method that
    sidesteps the NeMo typed-method kwargs check.
    """

    def __init__(self, nemo_joiner: nn.Module) -> None:
        super().__init__()
        self.inner = nemo_joiner

    def forward(
        self,
        enc_step: torch.Tensor,
        pred_emb: torch.Tensor,
    ) -> torch.Tensor:
        # joint(f, g) expects f=[B, d_model, T], g=[B, pred_hidden, U].
        # Caller is responsible for transposing inputs if they come in [B, T, C] form.
        return self.inner.joint(enc_step, pred_emb)
