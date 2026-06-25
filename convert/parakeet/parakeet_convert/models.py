"""PyTorch nn.Module wrappers for Parakeet TDT 0.6B → Core AI.

Differs from the Nemotron wrappers in two ways:
  1. The encoder is FULL-CONTEXT (no streaming caches) — so the wrapper has no
     register_buffer cache state and calls the plain `forward(audio_signal, length)`.
  2. The joint is TDT: its output is [B, T, U, vocab+1 + num_durations] — token
     logits (8193) followed by duration logits (5). The Swift TDT decoder splits
     them and advances the encoder time index by the predicted duration.

Shapes (Parakeet TDT 0.6B, probed from the loaded model):
  encoder(audio_signal=[1,128,T], length=[1])     -> enc_out [1, 1024, T_out]
  prediction.predict(y=[1,1] int32, [h,c])        -> pred_emb [1, 1, 640]
  joint.joint(enc=[1,1,1024], pred=[1,1,640])      -> logits [1, 1, 1, 8198]
                                                       (8193 token + 5 duration)
"""

from __future__ import annotations

import torch
from torch import nn


class ParakeetEncoderForCoreAI(nn.Module):
    """Full-context FastConformer encoder — no streaming caches."""

    def __init__(self, nemo_encoder: nn.Module) -> None:
        super().__init__()
        self.inner = nemo_encoder

    def forward(self, mel: torch.Tensor, mel_length: torch.Tensor) -> torch.Tensor:
        # Full-context forward: no cache tensors. Returns (encoded, encoded_len);
        # encoded is channels-second [B, d_model, T_out].
        encoded, _enc_len = self.inner(audio_signal=mel, length=mel_length)
        # Transpose to channels-last [B, T_out, d_model] so the Swift decoder can
        # iterate t over shape[1] (same convention as the Nemotron path).
        return encoded.transpose(1, 2).contiguous()


class ParakeetPredictionForCoreAI(nn.Module):
    """Single-step RNN-T prediction LSTM (identical shape to Nemotron's)."""

    def __init__(
        self,
        nemo_prediction: nn.Module,
        pred_hidden: int = 640,
        n_lstm_layers: int = 2,
    ) -> None:
        super().__init__()
        self.inner = nemo_prediction
        self.register_buffer(
            "lstm_h", torch.zeros(n_lstm_layers, 1, pred_hidden, dtype=torch.float32)
        )
        self.register_buffer(
            "lstm_c", torch.zeros(n_lstm_layers, 1, pred_hidden, dtype=torch.float32)
        )

    def forward(self, prev_token: torch.Tensor) -> torch.Tensor:
        if prev_token.dim() == 1:
            prev_token = prev_token.unsqueeze(0)
        pred_emb, new_state = self.inner.predict(
            prev_token, [self.lstm_h, self.lstm_c], False, 1
        )
        self.lstm_h.copy_(new_state[0])
        self.lstm_c.copy_(new_state[1])
        return pred_emb


class ParakeetJoinerForCoreAI(nn.Module):
    """TDT joint — output includes the 5 duration logits after the token logits."""

    def __init__(self, nemo_joiner: nn.Module) -> None:
        super().__init__()
        self.inner = nemo_joiner

    def forward(self, enc_step: torch.Tensor, pred_emb: torch.Tensor) -> torch.Tensor:
        # joint(f, g) -> [B, T, U, vocab+1+num_durations]. The TDT joint already
        # concatenates the duration logits; the Swift decoder splits them.
        return self.inner.joint(enc_step, pred_emb)
