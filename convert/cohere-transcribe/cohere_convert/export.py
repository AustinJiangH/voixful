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


def encoder_example(mel_frames: int = MEL_FRAMES):
    mel = torch.zeros(1, N_MELS, mel_frames, dtype=torch.float32)
    length = torch.tensor([mel_frames], dtype=torch.int32)
    return mel, length


def trace_t_enc(encoder_wrapper, mel_frames: int = MEL_FRAMES) -> int:
    """Run the encoder once to learn its fixed output time length (T_enc)."""
    mel, length = encoder_example(mel_frames)
    with torch.no_grad():
        out = encoder_wrapper(mel, length)
    return out.shape[1]


def export_encoder(encoder_wrapper, mel_frames: int = MEL_FRAMES):
    ep = torch.export.export(encoder_wrapper, encoder_example(mel_frames))
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


def export_cross_kv_projector(projector_wrapper, t_enc: int):
    example = (torch.zeros(1, t_enc, DEC_HIDDEN, dtype=torch.float32),)
    ep = torch.export.export(projector_wrapper, example)
    return ep.run_decompositions(coreai_torch.get_decomp_table())


def cached_decoder_example(t_enc: int):
    return (
        torch.zeros(1, 1, dtype=torch.int32),                  # input_id
        torch.zeros(1, 1, dtype=torch.int32),                  # position_id
        torch.zeros(1, 1, 1, CACHE_LEN, dtype=torch.float32),  # self_attention_mask
        torch.zeros(1, 1, 1, t_enc, dtype=torch.float32),      # cross_attention_mask
        torch.zeros(N_LAYERS, 1, N_HEADS, t_enc, HEAD_DIM, dtype=torch.float32),  # cross_k
        torch.zeros(N_LAYERS, 1, N_HEADS, t_enc, HEAD_DIM, dtype=torch.float32),  # cross_v
    )


def export_cached_decoder(decoder_wrapper, t_enc: int):
    ep = torch.export.export(decoder_wrapper, cached_decoder_example(t_enc))
    return ep.run_decompositions(coreai_torch.get_decomp_table())


def fused_decoder_example(t_enc: int):
    return (
        torch.zeros(1, 1, dtype=torch.int32),   # input_id
        torch.zeros(1, 1, dtype=torch.int32),   # position_id
        torch.tensor([t_enc], dtype=torch.int32),  # enc_len
        torch.zeros(N_LAYERS, 1, N_HEADS, t_enc, HEAD_DIM, dtype=torch.float32),  # cross_k
        torch.zeros(N_LAYERS, 1, N_HEADS, t_enc, HEAD_DIM, dtype=torch.float32),  # cross_v
    )


def export_fused_decoder(decoder_wrapper, t_enc: int):
    ep = torch.export.export(decoder_wrapper, fused_decoder_example(t_enc))
    return ep.run_decompositions(coreai_torch.get_decomp_table())


def export_x4_decoder(decoder_wrapper, t_enc: int):
    # Same input signature as the fused decoder; output is a [1,4] token block.
    ep = torch.export.export(decoder_wrapper, fused_decoder_example(t_enc))
    return ep.run_decompositions(coreai_torch.get_decomp_table())
