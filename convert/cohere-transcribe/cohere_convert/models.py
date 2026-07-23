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


def _fused_qkv(layers) -> nn.ModuleList:
    """One [hidden -> 3*hidden] Linear per layer replacing the three separate
    self-attention q/k/v projections — 1 matmul dispatch instead of 3. Weights
    are concatenated copies; the originals go uncaptured by torch.export since
    forward no longer touches them, so asset size is unchanged."""
    fused = nn.ModuleList()
    for layer in layers:
        a = layer.first_sub_layer
        w = torch.cat([a.query_net.weight, a.key_net.weight, a.value_net.weight], dim=0)
        has_bias = a.query_net.bias is not None
        lin = nn.Linear(w.shape[1], w.shape[0], bias=has_bias)
        with torch.no_grad():
            lin.weight.copy_(w)
            if has_bias:
                lin.bias.copy_(
                    torch.cat([a.query_net.bias, a.key_net.bias, a.value_net.bias], dim=0)
                )
        fused.append(lin)
    return fused


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
        self.qkv = _fused_qkv(self.layers)

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
        # `oh` stays a [1,1,M,1] broadcast — coreai-torch 0.4.1 converts stride-0
        # broadcast multiplies correctly (toy-verified bit-exact; the 0.4.0 leak
        # that forced full .repeat() materialization is fixed), so each cache
        # write is one fused elementwise kernel instead of repeat+multiply over
        # a materialized [1,H,M,D].
        idx = torch.arange(self.max_cache_len, device=h.device).view(1, -1)  # [1,M]
        sel = (idx == position_id.view(-1, 1)).to(h.dtype)                   # [1,M]
        oh = sel.view(1, 1, self.max_cache_len, 1)                           # [1,1,M,1]

        new_k, new_v = [], []
        for i, layer in enumerate(self.layers):
            # --- self-attention (external cache) ---
            residual = h
            hn = layer.layer_norm_1(h)
            a = layer.first_sub_layer
            qkv = self.qkv[i](hn)                 # [1,1,3*hidden], 1 matmul
            qh, kh, vh = qkv.chunk(3, dim=-1)
            q = self._heads(a, qh)                # [1,H,1,D]
            nk = self._heads(a, kh)               # [1,H,1,D]
            nv = self._heads(a, vh)
            kc = k_cache[i] * (1.0 - oh) + nk * oh  # [1,H,M,D] via broadcast
            vc = v_cache[i] * (1.0 - oh) + nv * oh
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


class CohereCrossKVProjectorForCoreAI(nn.Module):
    """Project encoder output to every layer's cross-attention K/V ONCE per
    utterance.

    The stateless `decode_step` recomputes these projections every token —
    8 layers x 2 x matmul([T_enc,1024]x[1024,1024]) per step, roughly half the
    per-step FLOPs. Running them once and feeding the result to
    `decode_step_cached` removes the recompute. Plain outputs (not states):
    torch.export bakes READ-ONLY buffers as constants, so the cached decoder
    takes these as regular inputs instead.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        dec = model.transf_decoder
        self.layers = dec._decoder.layers

    def _heads(self, attn, x):
        b, t, _ = x.shape
        return x.view(b, t, attn.num_heads, attn.head_dim).transpose(1, 2)

    def forward(self, encoder_hidden_states: torch.Tensor):  # [1, T_enc, 1024]
        ks, vs = [], []
        for layer in self.layers:
            c = layer.second_sub_layer
            ks.append(self._heads(c, c.key_net(encoder_hidden_states)))
            vs.append(self._heads(c, c.value_net(encoder_hidden_states)))
        return torch.stack(ks, dim=0), torch.stack(vs, dim=0)  # [L,1,H,T_enc,D] x2


class CohereDecoderStepCachedForCoreAI(nn.Module):
    """One decoder step with a STATEFUL self-attention cache and precomputed
    cross-attention K/V.

    vs the stateless `decode_step`: cross-attention K/V come precomputed from
    `project_cross_kv` (not re-projected from the 1.8 MB encoder output every
    step), and the self-attention cache is a mutated Core AI state (in-place
    MutableViews) instead of being threaded in/out (~14 MB/token). cross_k/v
    are regular INPUTS — torch.export bakes read-only buffers as constants, so
    they cannot be states here.

    Same one-hot cache-write and `.repeat()` materialization as `decode_step` —
    coreai_torch mis-converts stride-0 broadcast multiplies (see that class).
    """

    def __init__(self, model: nn.Module, max_cache_len: int = 108):
        super().__init__()
        dec = model.transf_decoder
        self.embedding = dec._embedding
        self.layers = dec._decoder.layers
        self.final_layer_norm = dec._decoder.final_layer_norm
        self.head = model.log_softmax
        first_attn = self.layers[0].first_sub_layer
        self.num_heads = first_attn.num_heads
        self.head_dim = first_attn.head_dim
        self.hidden_size = first_attn.hidden_size
        self.n_layers = len(self.layers)
        self.max_cache_len = max_cache_len
        self.register_buffer(
            "k_cache",
            torch.zeros(self.n_layers, 1, self.num_heads, max_cache_len, self.head_dim),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(self.n_layers, 1, self.num_heads, max_cache_len, self.head_dim),
        )
        self.qkv = _fused_qkv(self.layers)

    def _heads(self, attn, x):
        b, t, _ = x.shape
        return x.view(b, t, attn.num_heads, attn.head_dim).transpose(1, 2)

    def _merge(self, x):
        b, _, t, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, self.hidden_size)

    @staticmethod
    def _attention(q, k, v, attn_mask, scale):
        # Manual SDPA — F.scaled_dot_product_attention mis-converts (see
        # CohereDecoderStepForCoreAI._attention).
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale + attn_mask
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def forward(
        self,
        input_id: torch.Tensor,             # [1,1] int
        position_id: torch.Tensor,          # [1,1] int
        self_attention_mask: torch.Tensor,  # [1,1,1,M] additive
        cross_attention_mask: torch.Tensor, # [1,1,1,T_enc] additive
        cross_k: torch.Tensor,              # [L,1,H,T_enc,D] from project_cross_kv
        cross_v: torch.Tensor,              # [L,1,H,T_enc,D]
    ):
        input_id = input_id.long()
        position_id = position_id.long()
        h = self.embedding(input_id, position_id)  # [1,1,1024]

        idx = torch.arange(self.max_cache_len, device=h.device).view(1, -1)
        sel = (idx == position_id.view(-1, 1)).to(h.dtype)
        oh = sel.view(1, 1, self.max_cache_len, 1)  # broadcast; see stateless class

        new_k, new_v = [], []
        for i, layer in enumerate(self.layers):
            # --- self-attention (state cache) ---
            residual = h
            hn = layer.layer_norm_1(h)
            a = layer.first_sub_layer
            qkv = self.qkv[i](hn)
            qh, kh, vh = qkv.chunk(3, dim=-1)
            q = self._heads(a, qh)
            nk = self._heads(a, kh)
            nv = self._heads(a, vh)
            kc = self.k_cache[i] * (1.0 - oh) + nk * oh
            vc = self.v_cache[i] * (1.0 - oh) + nv * oh
            new_k.append(kc)
            new_v.append(vc)
            attn = self._attention(q, kc, vc, self_attention_mask, a.scale)
            h = residual + a.out_projection(self._merge(attn))

            # --- cross-attention (precomputed K/V) ---
            residual = h
            hn = layer.layer_norm_2(h)
            c = layer.second_sub_layer
            qx = self._heads(c, c.query_net(hn))
            cross = self._attention(
                qx, cross_k[i], cross_v[i], cross_attention_mask, c.scale
            )
            h = residual + c.out_projection(self._merge(cross))

            # --- feed-forward ---
            residual = h
            h = residual + layer.third_sub_layer(layer.layer_norm_3(h))

        # Write the updated self-attention cache back into state, in place.
        self.k_cache.copy_(torch.stack(new_k, dim=0))
        self.v_cache.copy_(torch.stack(new_v, dim=0))

        h = self.final_layer_norm(h)
        logits = self.head.mlp.layer0(h)  # [1,1,16384]
        return logits


class CohereDecoderX4ForCoreAI(CohereDecoderStepCachedForCoreAI):
    """FOUR decode steps statically unrolled in one graph. Per-step host
    dispatch (~4.5 ms/call, the measured floor) dominates the decoder, so the
    argmax of each sub-step feeds the next sub-step's embedding IN-GRAPH and
    the self-attention cache state is read/written once per block of four.
    The host consumes the [1,4] token block and truncates at EOS (sub-steps
    after EOS compute garbage that is never used). 108 cache slots = exactly
    27 blocks.
    """

    UNROLL = 4

    def __init__(self, model: nn.Module, t_enc: int, max_cache_len: int = 108):
        super().__init__(model, max_cache_len=max_cache_len)
        self.t_enc = t_enc

    def _one_step(self, cur_id, pos, cross_k, cross_v, cross_mask, k_all, v_all):
        """One decoder layer-stack pass against local cache tensors; returns
        (logits, new_k_all, new_v_all). Mirrors CohereDecoderStepCachedForCoreAI
        but with the cache as SSA locals so sub-steps chain inside the graph."""
        sidx = torch.arange(self.max_cache_len).view(1, -1)
        neg = torch.tensor(-1e9, dtype=torch.float32)
        zero = torch.tensor(0.0, dtype=torch.float32)
        self_mask = torch.where(sidx <= pos, zero, neg).view(1, 1, 1, self.max_cache_len)

        h = self.embedding(cur_id, pos.view(1, 1))
        sel = (sidx == pos).to(h.dtype)
        oh = sel.view(1, 1, self.max_cache_len, 1)  # broadcast; see stateless class

        new_k, new_v = [], []
        for i, layer in enumerate(self.layers):
            residual = h
            hn = layer.layer_norm_1(h)
            a = layer.first_sub_layer
            qkv = self.qkv[i](hn)
            qh, kh, vh = qkv.chunk(3, dim=-1)
            q = self._heads(a, qh)
            nk = self._heads(a, kh)
            nv = self._heads(a, vh)
            kc = k_all[i] * (1.0 - oh) + nk * oh
            vc = v_all[i] * (1.0 - oh) + nv * oh
            new_k.append(kc)
            new_v.append(vc)
            attn = self._attention(q, kc, vc, self_mask, a.scale)
            h = residual + a.out_projection(self._merge(attn))

            residual = h
            hn = layer.layer_norm_2(h)
            c = layer.second_sub_layer
            qx = self._heads(c, c.query_net(hn))
            cross = self._attention(qx, cross_k[i], cross_v[i], cross_mask, c.scale)
            h = residual + c.out_projection(self._merge(cross))

            residual = h
            h = residual + layer.third_sub_layer(layer.layer_norm_3(h))

        h = self.final_layer_norm(h)
        logits = self.head.mlp.layer0(h)
        return logits, torch.stack(new_k, dim=0), torch.stack(new_v, dim=0)

    def forward(  # type: ignore[override]
        self,
        input_id: torch.Tensor,   # [1,1] int — first sub-step's input token
        position_id: torch.Tensor,  # [1,1] int — first sub-step's position
        enc_len: torch.Tensor,    # [1] int
        cross_k: torch.Tensor,    # [L,1,H,T_enc,D]
        cross_v: torch.Tensor,    # [L,1,H,T_enc,D]
    ):
        neg = torch.tensor(-1e9, dtype=torch.float32)
        zero = torch.tensor(0.0, dtype=torch.float32)
        cidx = torch.arange(self.t_enc).view(1, -1)
        cross_mask = torch.where(cidx < enc_len.long().view(-1, 1), zero, neg).view(
            1, 1, 1, self.t_enc
        )

        cur = input_id.long()
        base = position_id.long().view(-1, 1)
        k_all = self.k_cache
        v_all = self.v_cache
        toks = []
        for s in range(self.UNROLL):
            pos = base + s  # [1,1]
            logits, k_all, v_all = self._one_step(
                cur, pos, cross_k, cross_v, cross_mask, k_all, v_all
            )
            cur = torch.argmax(logits, dim=-1)  # [1,1] int64, feeds next embedding
            toks.append(cur.to(torch.int32))

        self.k_cache.copy_(k_all)
        self.v_cache.copy_(v_all)
        return torch.cat(toks, dim=1)  # [1,4] int32


class CohereDecoderStepFusedForCoreAI(CohereDecoderStepCachedForCoreAI):
    """`decode_step_cached` with the last host round-trip costs folded in-graph:
    both attention masks are built from `position_id` / `enc_len` (the same
    arange-compare pattern as the one-hot cache write), and the output is the
    argmax TOKEN [1,1] int32 instead of 16384 logits. Per-step host I/O shrinks
    to three scalars in, one scalar out.
    """

    def __init__(self, model: nn.Module, t_enc: int, max_cache_len: int = 108):
        super().__init__(model, max_cache_len=max_cache_len)
        self.t_enc = t_enc

    def forward(  # type: ignore[override]
        self,
        input_id: torch.Tensor,   # [1,1] int
        position_id: torch.Tensor,  # [1,1] int
        enc_len: torch.Tensor,    # [1] int (valid encoder frames)
        cross_k: torch.Tensor,    # [L,1,H,T_enc,D]
        cross_v: torch.Tensor,    # [L,1,H,T_enc,D]
    ):
        neg = torch.tensor(-1e9, dtype=torch.float32)
        zero = torch.tensor(0.0, dtype=torch.float32)
        pos = position_id.long().view(-1, 1)
        sidx = torch.arange(self.max_cache_len).view(1, -1)
        self_mask = torch.where(sidx <= pos, zero, neg).view(1, 1, 1, self.max_cache_len)
        cidx = torch.arange(self.t_enc).view(1, -1)
        cross_mask = torch.where(cidx < enc_len.long().view(-1, 1), zero, neg).view(
            1, 1, 1, self.t_enc
        )
        logits = super().forward(
            input_id, position_id, self_mask, cross_mask, cross_k, cross_v
        )
        return torch.argmax(logits, dim=-1).to(torch.int32)  # [1,1]
