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

        Plain 4-D GQA: repeat K/V per query-head group. The reshape-grouped
        5-D formulation mis-computes under the Core AI runtime (row
        scrambling through the non-contiguous reshape)."""
        rep = self.n_heads // self.n_kv
        T = q.shape[2]
        kr = k.repeat_interleave(rep, dim=1)
        vr = v.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, kr.transpose(-1, -2)) / (self.head_dim ** 0.5)
        scores = scores + mask
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(probs, vr)
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
        c = self.core
        shape = (c.n_layers, 1, c.n_kv, MAX_SEQ, c.head_dim)
        self.register_buffer("k_cache", torch.zeros(shape))
        self.register_buffer("v_cache", torch.zeros(shape))
        causal = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min)
        self.register_buffer("causal_mask", torch.triu(causal, diagonal=1).view(1, 1, seq_len, seq_len))

    def forward(self, embeds: torch.Tensor, last_idx: torch.Tensor) -> torch.Tensor:
        c = self.core
        L = self.seq_len
        cos = c.rope_cos[:L]
        sin = c.rope_sin[:L]
        x = embeds
        for li, layer in enumerate(c.model.layers):
            h = _rms_norm(x, layer.input_layernorm.weight, c.eps)
            q, k, v = c.attn_qkv(layer, h, cos, sin)
            self.k_cache[li, :, :, :L] = k[0]
            self.v_cache[li, :, :, :L] = v[0]
            att = c.attend(layer, q, k, v, self.causal_mask)
            x = x + att
            h = _rms_norm(x, layer.post_attention_layernorm.weight, c.eps)
            x = x + c.mlp(layer, h)
        x = _rms_norm(x, c.model.norm.weight, c.eps)
        # index_select with a runtime index mis-lowers (reads a pad row);
        # one-hot matmul extraction is exact.
        onehot = (torch.arange(L, dtype=torch.int32) == last_idx.reshape(1)).to(x.dtype)
        last = torch.matmul(onehot.view(1, 1, L), x).squeeze(1)      # [1,H]
        logits = c.lm_head(last)
        return torch.argmax(logits, dim=-1).to(torch.int32)


class QwenStepForCoreAI(nn.Module):
    """One AR step: token at position pos; one-hot cache write; mask idx<=pos."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.core = _QwenCore(hf_model)
        self.embed = hf_model.model.embed_tokens
        c = self.core
        shape = (c.n_layers, 1, c.n_kv, MAX_SEQ, c.head_dim)
        self.register_buffer("k_cache", torch.zeros(shape))
        self.register_buffer("v_cache", torch.zeros(shape))
        self.register_buffer("arangeS", torch.arange(MAX_SEQ, dtype=torch.int32))

    def forward(self, tok: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
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
        for li, layer in enumerate(c.model.layers):
            h = _rms_norm(x, layer.input_layernorm.weight, c.eps)
            q, k, v = c.attn_qkv(layer, h, cos, sin)       # q/k/v: [1,·,1,D]
            # Broadcast one-hot write at row `pos` (no scatter, no .repeat()):
            # k*onehot is [1,KV,MAX,D] with k's row at pos, zeros elsewhere.
            self.k_cache[li] = self.k_cache[li] * (1 - onehot[0]) + (k * onehot)[0]
            self.v_cache[li] = self.v_cache[li] * (1 - onehot[0]) + (v * onehot)[0]
            att = c.attend(layer, q, self.k_cache[li], self.v_cache[li], keymask)
            x = x + att
            h = _rms_norm(x, layer.post_attention_layernorm.weight, c.eps)
            x = x + c.mlp(layer, h)
        x = _rms_norm(x, c.model.norm.weight, c.eps)
        logits = c.lm_head(x.squeeze(1))
        return torch.argmax(logits, dim=-1).to(torch.int32)


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


class QwenPrefillNoWrite(QwenPrefillForCoreAI):
    """Diagnostic: identical compute, NO cache writes."""

    def forward(self, embeds, last_idx):  # noqa: D102
        c = self.core
        L = self.seq_len
        cos = c.rope_cos[:L]
        sin = c.rope_sin[:L]
        x = embeds
        for li, layer in enumerate(c.model.layers):
            h = _rms_norm(x, layer.input_layernorm.weight, c.eps)
            q, k, v = c.attn_qkv(layer, h, cos, sin)
            att = c.attend(layer, q, k, v, self.causal_mask)
            x = x + att
            h = _rms_norm(x, layer.post_attention_layernorm.weight, c.eps)
            x = x + c.mlp(layer, h)
        x = _rms_norm(x, c.model.norm.weight, c.eps)
        onehot = (torch.arange(L, dtype=torch.int32) == last_idx.reshape(1)).to(x.dtype)
        last = torch.matmul(onehot.view(1, 1, L), x).squeeze(1)
        return torch.argmax(c.lm_head(last), dim=-1).to(torch.int32)


class QwenPrefillFunctional(QwenPrefillForCoreAI):
    """Caches as PLAIN OUTPUTS (no states): host threads them to the step."""

    def forward(self, embeds, last_idx):  # noqa: D102
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
        x = _rms_norm(x, c.model.norm.weight, c.eps)
        onehot = (torch.arange(L, dtype=torch.int32) == last_idx.reshape(1)).to(x.dtype)
        last = torch.matmul(onehot.view(1, 1, L), x).squeeze(1)
        tok = torch.argmax(c.lm_head(last), dim=-1).to(torch.int32)
        k_out = torch.cat(new_k, dim=0)   # [28,KV,L,D]
        v_out = torch.cat(new_v, dim=0)
        return tok, k_out, v_out


class QwenStepFunctional(nn.Module):
    """One AR step, fully functional: caches in as inputs, out as outputs.
    Prompt caches [28,1,KV,L,D] stay FIXED inputs; generated-token caches
    [28,1,KV,G,D] grow by concat (host passes the accumulated tail)."""

    def __init__(self, hf_model: nn.Module, prompt_len: int, gen_cap: int):
        super().__init__()
        self.core = _QwenCore(hf_model)
        self.embed = hf_model.model.embed_tokens
        self.prompt_len = prompt_len
        self.gen_cap = gen_cap
        self.register_buffer("arangeG", torch.arange(gen_cap, dtype=torch.int32))

    def forward(self, tok, pos, k_prompt, v_prompt, k_gen, v_gen):
        # pos = absolute position of THIS token; gen index g = pos - prompt_len
        c = self.core
        p = pos.to(torch.int64).reshape(1)
        g = (pos - self.prompt_len).reshape(1)
        x = self.embed(tok.to(torch.int64)).view(1, 1, c.hidden)
        cos = torch.nn.functional.embedding(p, c.rope_cos)
        sin = torch.nn.functional.embedding(p, c.rope_sin)
        onehot = (self.arangeG == g).to(x.dtype).view(1, 1, self.gen_cap, 1)
        # keys valid: all prompt rows + gen rows <= g
        genmask = torch.where(
            self.arangeG.view(1, 1, 1, self.gen_cap) <= g.reshape(1, 1, 1, 1),
            torch.zeros(1, dtype=x.dtype),
            torch.full((1,), torch.finfo(torch.float32).min, dtype=x.dtype),
        )
        mask = torch.cat(
            [torch.zeros(1, 1, 1, self.prompt_len, dtype=x.dtype), genmask], dim=-1)
        new_kg, new_vg = [], []
        for li, layer in enumerate(c.model.layers):
            h = _rms_norm(x, layer.input_layernorm.weight, c.eps)
            q, k, v = c.attn_qkv(layer, h, cos, sin)
            kg = k_gen[li] * (1 - onehot[0]) + (k * onehot)[0]
            vg = v_gen[li] * (1 - onehot[0]) + (v * onehot)[0]
            new_kg.append(kg)
            new_vg.append(vg)
            kf = torch.cat([k_prompt[li], kg], dim=1).unsqueeze(0)  # [1,KV,L+G,D]
            vf = torch.cat([v_prompt[li], vg], dim=1).unsqueeze(0)
            att = c.attend(layer, q, kf, vf, mask)
            x = x + att
            h = _rms_norm(x, layer.post_attention_layernorm.weight, c.eps)
            x = x + c.mlp(layer, h)
        x = _rms_norm(x, c.model.norm.weight, c.eps)
        tok_out = torch.argmax(c.lm_head(x.squeeze(1)), dim=-1).to(torch.int32)
        return tok_out, torch.stack(new_kg, dim=0), torch.stack(new_vg, dim=0)
