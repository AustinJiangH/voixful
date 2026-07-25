"""Core AI export wrappers for OpenAI Whisper (attention encoder-decoder).

Two functions, same shape as the Cohere Transcribe pipeline:

  whisper_encode(input_features[1, n_mels, 3000]) -> hidden_states[1, T_enc, d]
  whisper_decode(input_id, position_id, encoder_hidden_states,
                 self_attention_mask, k_cache[L,1,H,M,D], v_cache[L,1,H,M,D])
      -> logits[1,1,vocab], k_cache_out, v_cache_out

The decoder uses an EXTERNAL KV cache: self-attention K/V are threaded in/out as
plain tensors (written at `position_id` via a one-hot, so no dynamic scatter);
cross-attention K/V are recomputed each step from `encoder_hidden_states`. The
host builds the causal self-attention mask. Whisper cross-attention needs no
mask — the encoder always sees the full 30 s window, so every encoder frame is
valid. Both functions are stateless (no Core AI MutableBuffers) so coreai_torch
can carry them, mirroring Cohere's legacy `decode_step`.

Attention is spelled out manually (not F.scaled_dot_product_attention): the
Cohere pipeline found coreai_torch mis-converts SDPA, so we use the same
matmul->add-mask->softmax->matmul form here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class WhisperEncoderForCoreAI(nn.Module):
    """The Whisper audio encoder: 2 conv frontend + transformer stack, emitting
    `[1, T_enc, d]` (T_enc = 1500 for the 30 s / 3000-frame mel window)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_features).last_hidden_state  # [1, T_enc, d]


class WhisperDecoderStepForCoreAI(nn.Module):
    """One autoregressive decoder step with an external self-attention KV cache."""

    def __init__(self, model: nn.Module, max_cache_len: int):
        super().__init__()
        dec = model.model.decoder
        self.embed_tokens = dec.embed_tokens
        self.embed_positions = dec.embed_positions  # nn.Embedding-like; .weight [P, d]
        self.layers = dec.layers
        self.final_layer_norm = dec.layer_norm
        self.proj_out = model.proj_out  # [d -> vocab], tied to embed_tokens

        first = self.layers[0].self_attn
        self.num_heads = first.num_heads
        self.head_dim = first.head_dim
        self.hidden_size = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5
        self.n_layers = len(self.layers)
        self.max_cache_len = max_cache_len

    def _heads(self, x):
        # [1, t, hidden] -> [1, heads, t, head_dim]
        b, t, _ = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x):
        # [1, heads, t, head_dim] -> [1, t, hidden]
        b, _, t, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, self.hidden_size)

    @staticmethod
    def _attention(q, k, v, attn_mask=None):
        # Manual scaled-dot-product attention (q is pre-scaled). additive mask
        # or None. coreai_torch mis-converts F.scaled_dot_product_attention.
        scores = torch.matmul(q, k.transpose(-2, -1))
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def forward(
        self,
        input_id: torch.Tensor,             # [1,1] int
        position_id: torch.Tensor,          # [1,1] int (current decode step)
        encoder_hidden_states: torch.Tensor,   # [1, T_enc, d]
        self_attention_mask: torch.Tensor,     # [1,1,1,M] additive (-inf future)
        k_cache: torch.Tensor,              # [L,1,H,M,D]
        v_cache: torch.Tensor,              # [L,1,H,M,D]
    ):
        input_id = input_id.long()
        position_id = position_id.long()
        # token embedding + learned positional embedding at `position_id`.
        h = self.embed_tokens(input_id) + self.embed_positions.weight[position_id.view(-1)].view(1, 1, -1)

        # One-hot cache write at `position_id` (export-friendly, no scatter).
        idx = torch.arange(self.max_cache_len, device=h.device).view(1, -1)  # [1,M]
        sel = (idx == position_id.view(-1, 1)).to(h.dtype)                   # [1,M]
        oh = sel.view(1, 1, self.max_cache_len, 1)                          # [1,1,M,1]

        new_k, new_v = [], []
        for i, layer in enumerate(self.layers):
            # --- self-attention (external cache) ---
            residual = h
            hn = layer.self_attn_layer_norm(h)
            a = layer.self_attn
            q = self._heads(a.q_proj(hn) * self.scale)   # [1,H,1,D]
            nk = self._heads(a.k_proj(hn))               # [1,H,1,D]
            nv = self._heads(a.v_proj(hn))
            kc = k_cache[i] * (1.0 - oh) + nk * oh        # [1,H,M,D] via broadcast
            vc = v_cache[i] * (1.0 - oh) + nv * oh
            new_k.append(kc)
            new_v.append(vc)
            attn = self._attention(q, kc, vc, self_attention_mask)
            h = residual + a.out_proj(self._merge(attn))

            # --- cross-attention (recomputed from encoder each step, no mask) ---
            residual = h
            hn = layer.encoder_attn_layer_norm(h)
            c = layer.encoder_attn
            qx = self._heads(c.q_proj(hn) * self.scale)                # [1,H,1,D]
            kx = self._heads(c.k_proj(encoder_hidden_states))          # [1,H,T_enc,D]
            vx = self._heads(c.v_proj(encoder_hidden_states))
            cross = self._attention(qx, kx, vx, None)
            h = residual + c.out_proj(self._merge(cross))

            # --- feed-forward ---
            residual = h
            hn = layer.final_layer_norm(h)
            h = residual + layer.fc2(F.gelu(layer.fc1(hn)))

        h = self.final_layer_norm(h)
        logits = self.proj_out(h)  # [1,1,vocab]
        return logits, torch.stack(new_k, dim=0), torch.stack(new_v, dim=0)
