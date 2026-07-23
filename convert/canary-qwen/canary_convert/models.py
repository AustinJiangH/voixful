"""torch.export-friendly wrappers for Canary-Qwen 2.5B (SALM).

Split (mirrors the runtime call sequence):

  encode_<T>      : mel [1,128,T] + length -> audio_embeds [1, T/8, 2048]
                    (NeMo ConformerEncoder ×32 + the SALM 1024->2048 linear)
  embed_<N>       : token ids [1,N] -> embeds [1,N,2048] (host composes the
                    ChatML prompt: pre_ids ++ audio ++ post_ids)
  prefill_{asr|llm}_<L> : embeds [1,L,2048], last_idx [1] -> first token [1]
                    (causal pass; writes K/V cache rows [0,L) into states)
  step_{asr|llm}  : tok [1], pos [1] -> next token [1]
                    (one AR step; one-hot cache write at pos, mask idx<=pos)

The `asr` variant is the LoRA-MERGED Qwen3 (transcription); `llm` is the
pristine Qwen3-1.7B-instruct (formatting pass). Cache layout per variant:
k/v [n_layers, 1, n_kv_heads, MAX_SEQ, head_dim] as function STATES — the
Swift engine owns the buffers and passes them to prefill + every step
(Cohere decode pattern).

Positioning trick that keeps everything static: generated tokens are written
at their TRUE positions (prompt_len + t), which progressively overwrite the
prefill bucket's padded-garbage rows; a `key <= pos` additive mask is then
exact at every step. Prefill needs only the standard causal mask because
bucket padding sits strictly AFTER the valid prompt.
"""

from __future__ import annotations

import torch
from torch import nn

MAX_SEQ = 1024


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * weight.float()).to(x.dtype)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [1, heads, T, D]; cos/sin: [T, D]
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class _QwenCore(nn.Module):
    """Shared pieces: config scalars, rope tables, per-layer module refs."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        cfg = hf_model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.hidden = cfg.hidden_size
        self.eps = cfg.rms_norm_eps
        self.model = hf_model.model          # Qwen3Model (embed_tokens, layers, norm)
        self.lm_head = hf_model.lm_head
        # RoPE table for all MAX_SEQ positions, laid out rotate-half style.
        theta = cfg.rope_theta
        inv = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(MAX_SEQ).float()
        freqs = torch.outer(t, inv)                       # [MAX, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)           # [MAX, D]
        self.register_buffer("rope_cos", emb.cos())
        self.register_buffer("rope_sin", emb.sin())

    def attn_qkv(self, layer, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """(q,k,v) for a full sequence x [1,T,H] with Qwen3 q/k head norms."""
        a = layer.self_attn
        T = x.shape[1]
        q = a.q_proj(x).view(1, T, self.n_heads, self.head_dim)
        k = a.k_proj(x).view(1, T, self.n_kv, self.head_dim)
        v = a.v_proj(x).view(1, T, self.n_kv, self.head_dim)
        q = _rms_norm(q, a.q_norm.weight, self.eps).transpose(1, 2)   # [1,H,T,D]
        k = _rms_norm(k, a.k_norm.weight, self.eps).transpose(1, 2)   # [1,KV,T,D]
        v = v.transpose(1, 2)
        q = _rope(q, cos, sin)
        k = _rope(k, cos, sin)
        return q, k, v

    def attend(self, layer, q, k, v, mask):
        """softmax(qk^T/sqrt d + mask) v -> o_proj. k/v are [1,KV,S,D].

        GQA without materializing repeated K/V: fold the per-KV-group query
        heads into the row axis of one matmul per KV head."""
        rep = self.n_heads // self.n_kv
        T = q.shape[2]
        qg = q.reshape(1, self.n_kv, rep * T, self.head_dim)
        scores = torch.matmul(qg, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        scores = scores.reshape(1, self.n_kv, rep, T, -1) + mask.unsqueeze(1)
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(probs.reshape(1, self.n_kv, rep * T, -1), v)
        out = out.reshape(1, self.n_heads, T, self.head_dim)
        out = out.transpose(1, 2).reshape(1, T, self.n_heads * self.head_dim)
        return layer.self_attn.o_proj(out)

    def mlp(self, layer, x):
        m = layer.mlp
        return m.down_proj(nn.functional.silu(m.gate_proj(x)) * m.up_proj(x))


class QwenEmbedForCoreAI(nn.Module):
    """ids [1,N] i32 -> embeds [1,N,2048]."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.embed = hf_model.model.embed_tokens

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed(ids.to(torch.int64))


class QwenPrefillForCoreAI(nn.Module):
    """Causal pass over embeds [1,L,2048]; writes cache rows [0,L); returns
    argmax token at last_idx. Cache buffers are the exported STATES."""

    def __init__(self, hf_model: nn.Module, seq_len: int):
        super().__init__()
        self.core = _QwenCore(hf_model)
        self.seq_len = seq_len
        causal = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min)
        self.register_buffer("causal_mask", torch.triu(causal, diagonal=1).view(1, 1, seq_len, seq_len))

    def forward(
        self,
        embeds: torch.Tensor,
        last_idx: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):
        c = self.core
        L = self.seq_len
        cos = c.rope_cos[:L]
        sin = c.rope_sin[:L]
        x = embeds
        new_k, new_v = [], []
        for li, layer in enumerate(c.model.layers):
            h = _rms_norm(x, layer.input_layernorm.weight, c.eps)
            q, k, v = c.attn_qkv(layer, h, cos, sin)
            new_k.append(k)
            new_v.append(v)
            att = c.attend(layer, q, k, v, self.causal_mask)
            x = x + att
            h = _rms_norm(x, layer.post_attention_layernorm.weight, c.eps)
            x = x + c.mlp(layer, h)
        # Functional state update (Cohere-proven): caches are forward args,
        # updated tensors are outputs; state_names pairs them at export.
        pad = torch.zeros(len(new_k), 1, c.n_kv, MAX_SEQ - L, c.head_dim)
        k_out = torch.cat([torch.cat(new_k, dim=0).unsqueeze(1), pad], dim=3)
        v_out = torch.cat([torch.cat(new_v, dim=0).unsqueeze(1), pad], dim=3)
        x = _rms_norm(x, c.model.norm.weight, c.eps)
        # One-hot row extraction — aten.index_select with a runtime index
        # mis-lowers (reads a pad row); a matmul against the one-hot is exact.
        onehot = (torch.arange(L, dtype=torch.int32) == last_idx.reshape(1)).to(x.dtype)
        last = torch.matmul(onehot.view(1, 1, L), x).squeeze(1)      # [1,H]
        logits = c.lm_head(last)
        return torch.argmax(logits, dim=-1).to(torch.int32), k_out, v_out


class QwenStepForCoreAI(nn.Module):
    """One AR step: token at position pos; one-hot cache write; mask idx<=pos."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.core = _QwenCore(hf_model)
        self.embed = hf_model.model.embed_tokens
        self.register_buffer("arangeS", torch.arange(MAX_SEQ, dtype=torch.int32))

    def forward(
        self,
        tok: torch.Tensor,
        pos: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):
        c = self.core
        p = pos.to(torch.int64).reshape(1)
        x = self.embed(tok.to(torch.int64)).view(1, 1, c.hidden)
        cos = torch.nn.functional.embedding(p, c.rope_cos)  # [1, D]
        sin = torch.nn.functional.embedding(p, c.rope_sin)
        onehot = (self.arangeS == pos.reshape(1)).to(x.dtype).view(1, 1, MAX_SEQ, 1)
        keymask = torch.where(
            self.arangeS.view(1, 1, 1, MAX_SEQ) <= pos.reshape(1, 1, 1, 1),
            torch.zeros(1, dtype=x.dtype),
            torch.full((1,), torch.finfo(torch.float32).min, dtype=x.dtype),
        )
        new_k, new_v = [], []
        for li, layer in enumerate(c.model.layers):
            h = _rms_norm(x, layer.input_layernorm.weight, c.eps)
            q, k, v = c.attn_qkv(layer, h, cos, sin)       # q/k/v: [1,·,1,D]
            # Broadcast one-hot write at row `pos` (no scatter, no .repeat());
            # computed as a LOCAL tensor so this layer attends to it, then all
            # 28 layers commit in one full-buffer copy_ after the loop.
            knl = k_cache[li] * (1 - onehot[0]) + (k * onehot)[0]
            vnl = v_cache[li] * (1 - onehot[0]) + (v * onehot)[0]
            new_k.append(knl)
            new_v.append(vnl)
            att = c.attend(layer, q, knl, vnl, keymask)
            x = x + att
            h = _rms_norm(x, layer.post_attention_layernorm.weight, c.eps)
            x = x + c.mlp(layer, h)
        k_out = torch.stack(new_k, dim=0)
        v_out = torch.stack(new_v, dim=0)
        x = _rms_norm(x, c.model.norm.weight, c.eps)
        logits = c.lm_head(x.squeeze(1))
        return torch.argmax(logits, dim=-1).to(torch.int32), k_out, v_out


class CanaryEncoderForCoreAI(nn.Module):
    """mel [1,128,T] + length -> audio embeds [1, T/8, 2048].

    NeMo ConformerEncoder (32L, d1024, full context — same class as the
    Parakeet export) followed by the SALM 1024->2048 projection.
    """

    def __init__(self, nemo_encoder: nn.Module, proj: nn.Module) -> None:
        super().__init__()
        self.inner = nemo_encoder
        self.proj = proj

    def forward(self, mel: torch.Tensor, mel_length: torch.Tensor) -> torch.Tensor:
        encoded, _len = self.inner(audio_signal=mel, length=mel_length)
        return self.proj(encoded.transpose(1, 2))
