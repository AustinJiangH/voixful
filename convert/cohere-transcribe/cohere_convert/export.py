"""torch.export the two Cohere wrappers at fixed shapes for coreai_torch.

coreai_torch 0.4.0 can't carry dynamic axes, so every shape is fixed:
  - encoder mel window: 3500 frames (35 s)
  - self-attn cache length: 108 (max decode steps)
  - encoder output length T_ENC: determined by tracing the encoder once
"""

from __future__ import annotations

import coreai_torch
import torch

MEL_FRAMES = 3500
N_MELS = 128
CACHE_LEN = 108          # max_seq_len for decode
N_LAYERS = 8
N_HEADS = 8
HEAD_DIM = 128
DEC_HIDDEN = 1024
VOCAB = 16384


def encoder_example():
    mel = torch.zeros(1, N_MELS, MEL_FRAMES, dtype=torch.float32)
    length = torch.tensor([MEL_FRAMES], dtype=torch.int32)
    return mel, length


def trace_t_enc(encoder_wrapper) -> int:
    """Run the encoder once to learn its fixed output time length (T_enc)."""
    mel, length = encoder_example()
    with torch.no_grad():
        out = encoder_wrapper(mel, length)
    return out.shape[1]


def export_encoder(encoder_wrapper):
    ep = torch.export.export(encoder_wrapper, encoder_example())
    return ep.run_decompositions(coreai_torch.get_decomp_table())


def decoder_example(t_enc: int):
    return (
        torch.zeros(1, 1, dtype=torch.int32),                         # input_id
        torch.zeros(1, 1, dtype=torch.int32),                         # position_id
        torch.zeros(1, t_enc, DEC_HIDDEN, dtype=torch.float32),       # encoder_hidden_states
        torch.zeros(1, 1, 1, CACHE_LEN, dtype=torch.float32),         # self_attention_mask
        torch.zeros(1, 1, 1, t_enc, dtype=torch.float32),             # cross_attention_mask
        torch.zeros(N_LAYERS, 1, N_HEADS, CACHE_LEN, HEAD_DIM, dtype=torch.float32),  # k_cache
        torch.zeros(N_LAYERS, 1, N_HEADS, CACHE_LEN, HEAD_DIM, dtype=torch.float32),  # v_cache
    )


def export_decoder(decoder_wrapper, t_enc: int):
    ep = torch.export.export(decoder_wrapper, decoder_example(t_enc))
    return ep.run_decompositions(coreai_torch.get_decomp_table())
