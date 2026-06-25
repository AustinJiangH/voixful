"""Core AI export wrappers for Cohere Transcribe 2B (attention encoder-decoder).

Two stateless functions, matching the validated upstream CoreML interface:

  encode(input_features[1,128,3500], length[1]) -> hidden_states[1, T_enc, 1024]
  decode_step(input_id, position_id, encoder_hidden_states,
              self_attention_mask, cross_attention_mask,
              k_cache[L,1,H,M,D], v_cache[L,1,H,M,D])
      -> logits[1,1,16384], k_cache_out, v_cache_out

The decoder uses an EXTERNAL KV cache: self-attention K/V are threaded in/out as
plain tensors (written at `position_id` via a one-hot, so no dynamic scatter);
cross-attention K/V are recomputed each step from `encoder_hidden_states`. The
host builds both additive attention masks and passes them in. This keeps both
functions stateless — no Core AI MutableBuffers — so coreai_torch can carry them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CohereEncoderForCoreAI(nn.Module):
    """Full-context Fast-Conformer encoder + the 1280->1024 decoder projection."""

    def __init__(self, encoder: nn.Module, proj: nn.Module | None):
        super().__init__()
        self.encoder = encoder
        self.proj = proj

    def forward(self, input_features: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        x, _ = self.encoder(input_features, length.to(torch.long))  # [1, T, 1280]
        if self.proj is not None:
            x = self.proj(x)  # [1, T, 1024]
        return x


class CohereDecoderStepForCoreAI(nn.Module):
    """One autoregressive decoder step with an external self-attention KV cache."""

    def __init__(self, model: nn.Module, max_cache_len: int = 108):
        super().__init__()
        dec = model.transf_decoder
        self.embedding = dec._embedding
        self.layers = dec._decoder.layers
        self.final_layer_norm = dec._decoder.final_layer_norm
        self.head = model.log_softmax  # TokenClassifierHead (tied to token embedding)
        first_attn = self.layers[0].first_sub_layer
        self.num_heads = first_attn.num_heads
        self.head_dim = first_attn.head_dim
        self.hidden_size = first_attn.hidden_size
        self.n_layers = len(self.layers)
        self.max_cache_len = max_cache_len

    def _heads(self, attn, x):
        # [1, t, hidden] -> [1, heads, t, head_dim]
        b, t, _ = x.shape
        return x.view(b, t, attn.num_heads, attn.head_dim).transpose(1, 2)

    def _merge(self, x):
        # [1, heads, t, head_dim] -> [1, t, hidden]
        b, _, t, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, self.hidden_size)

    @staticmethod
    def _attention(q, k, v, attn_mask, scale):
        # Manual scaled-dot-product attention. coreai_torch mis-converts
        # F.scaled_dot_product_attention (the decoder logits came out off by ~10
        # vs torch while the manual-attention encoder was bit-exact), so we spell
        # it out: scores = (q·kᵀ)·scale + additive_mask -> softmax -> ·v.
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale + attn_mask
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def forward(
        self,
        input_id: torch.Tensor,            # [1,1] int
        position_id: torch.Tensor,         # [1,1] int  (current decode step)
        encoder_hidden_states: torch.Tensor,   # [1, T_enc, 1024]
        self_attention_mask: torch.Tensor,     # [1,1,1,M] additive (-inf future)
        cross_attention_mask: torch.Tensor,    # [1,1,1,T_enc] additive (-inf padding)
        k_cache: torch.Tensor,             # [L,1,H,M,D]
        v_cache: torch.Tensor,             # [L,1,H,M,D]
    ):
        # Core AI feeds int32; embedding lookups / one_hot need int64 indices.
        input_id = input_id.long()
        position_id = position_id.long()
        h = self.embedding(input_id, position_id)  # [1,1,1024]

        # One-hot over the cache axis so we write new K/V at `position_id` without
        # a dynamic scatter (export-friendly): cache' = cache*(1-oh) + new*oh.
        idx = torch.arange(self.max_cache_len, device=h.device).view(1, -1)  # [1,M]
        sel = (idx == position_id.view(-1, 1)).to(h.dtype)                   # [1,M]
        # Materialize the selector to full [1,H,M,D] with repeat, NOT a stride-0
        # broadcast view: coreai_torch mis-converts broadcast multiplies of a
        # size-1 dim (the new K/V leaked into every cache slot, off by ~5).
        oh = sel.view(1, 1, self.max_cache_len, 1).repeat(1, self.num_heads, 1, self.head_dim)

        new_k, new_v = [], []
        for i, layer in enumerate(self.layers):
            # --- self-attention (external cache) ---
            residual = h
            hn = layer.layer_norm_1(h)
            a = layer.first_sub_layer
            q = self._heads(a, a.query_net(hn))   # [1,H,1,D]
            nk = self._heads(a, a.key_net(hn))    # [1,H,1,D]
            nv = self._heads(a, a.value_net(hn))
            nk_full = nk.repeat(1, 1, self.max_cache_len, 1)  # [1,H,M,D] (real copy)
            nv_full = nv.repeat(1, 1, self.max_cache_len, 1)
            kc = k_cache[i] * (1.0 - oh) + nk_full * oh
            vc = v_cache[i] * (1.0 - oh) + nv_full * oh
            new_k.append(kc)
            new_v.append(vc)
            attn = self._attention(q, kc, vc, self_attention_mask, a.scale)
            h = residual + a.out_projection(self._merge(attn))

            # --- cross-attention (recomputed from encoder each step) ---
            residual = h
            hn = layer.layer_norm_2(h)
            c = layer.second_sub_layer
            qx = self._heads(c, c.query_net(hn))                 # [1,H,1,D]
            kx = self._heads(c, c.key_net(encoder_hidden_states))   # [1,H,T_enc,D]
            vx = self._heads(c, c.value_net(encoder_hidden_states))
            cross = self._attention(qx, kx, vx, cross_attention_mask, c.scale)
            h = residual + c.out_projection(self._merge(cross))

            # --- feed-forward ---
            residual = h
            h = residual + layer.third_sub_layer(layer.layer_norm_3(h))

        h = self.final_layer_norm(h)
        # Raw logits (skip the head's log_softmax): argmax is identical but avoids
        # a numerically heavy 16384-class log_softmax that can flip a borderline
        # greedy decision under Core AI's fp kernels.
        logits = self.head.mlp.layer0(h)  # [1,1,16384]
        return logits, torch.stack(new_k, dim=0), torch.stack(new_v, dim=0)
