"""torch.export the two Whisper wrappers at fixed shapes for coreai_torch.

coreai_torch can't carry dynamic axes, so every shape is fixed:
  - encoder mel window: `mel_frames` (3000 = 30 s) x `n_mels` (80 or 128)
  - encoder output length T_enc: determined by tracing the encoder once (1500)
  - self-attn cache length M: `max_cache_len` (the decoder token cap)
"""

from __future__ import annotations

import coreai_torch
import torch


def encoder_example(n_mels: int, mel_frames: int):
    return (torch.zeros(1, n_mels, mel_frames, dtype=torch.float32),)


def trace_t_enc(encoder_wrapper, n_mels: int, mel_frames: int) -> int:
    """Run the encoder once to learn its fixed output time length (T_enc)."""
    with torch.no_grad():
        out = encoder_wrapper(*encoder_example(n_mels, mel_frames))
    return out.shape[1]


def export_encoder(encoder_wrapper, n_mels: int, mel_frames: int):
    ep = torch.export.export(encoder_wrapper, encoder_example(n_mels, mel_frames))
    return ep.run_decompositions(coreai_torch.get_decomp_table())


def decoder_example(t_enc: int, d_model: int, n_layers: int, n_heads: int,
                    head_dim: int, max_cache_len: int):
    return (
        torch.zeros(1, 1, dtype=torch.int32),                              # input_id
        torch.zeros(1, 1, dtype=torch.int32),                              # position_id
        torch.zeros(1, t_enc, d_model, dtype=torch.float32),               # encoder_hidden_states
        torch.zeros(1, 1, 1, max_cache_len, dtype=torch.float32),          # self_attention_mask
        torch.zeros(n_layers, 1, n_heads, max_cache_len, head_dim, dtype=torch.float32),  # k_cache
        torch.zeros(n_layers, 1, n_heads, max_cache_len, head_dim, dtype=torch.float32),  # v_cache
    )


def export_decoder(decoder_wrapper, t_enc: int, d_model: int, n_layers: int,
                   n_heads: int, head_dim: int, max_cache_len: int):
    ep = torch.export.export(
        decoder_wrapper,
        decoder_example(t_enc, d_model, n_layers, n_heads, head_dim, max_cache_len),
    )
    return ep.run_decompositions(coreai_torch.get_decomp_table())
