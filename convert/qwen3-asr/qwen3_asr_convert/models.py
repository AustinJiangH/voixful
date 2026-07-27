"""torch.export-friendly wrappers for Qwen3-ASR (SALM: audio encoder + Qwen3).

Split (mirrors the runtime call sequence, same shape as Canary-Qwen):

  encode_<W>      : mel [1,128,W] + length -> audio_embeds [1, S, 1024]
                    (static reimpl of the chunked Whisper-style audio tower;
                    S = (W//100)*13 audio tokens; block-local attention win 104)
  embed_<N>       : token ids [1,N] -> embeds [1,N,1024]
  prefill_asr_<L> : embeds [1,L,1024], last_idx [1] -> first token + K/V caches
  step_qwen3      : tok [1], pos [1] -> next token [1] (one AR step)

The decoder wrappers (_QwenCore / QwenStep / QwenEmbed / QwenPrefill) are the
Canary-Qwen SALM decode path, geometry read from the text config (Qwen3: 1024
hidden, 28 layers, 16 heads, 8 KV, head_dim 128, rope 1e6). KV cache is fp16
states so the AR decoder lands on the ANE (the runtime rejects fp32 MemRef
states); positioning trick keeps every graph static (see QwenStep).

The AUDIO encoder is a STATIC reimplementation of Qwen3ASRAudioEncoder: the
real forward chunks by 100 mel frames, zero-pads the tail chunk, and uses ragged
block-local attention (cu_seqlens) — none of which torch.export can trace. For a
fixed window W (multiple of 100) with the audio pre-padded to W, the chunk count
and the block-diagonal attention mask are STATIC constants, so the same weights
run as a clean graph. Validated numerically + end-to-end vs the real audio_tower.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MAX_SEQ = 1024

# KV cache STORAGE dtype (state I/O boundary); compute stays fp32. fp16 lets the
# AR decoder land on the ANE (the runtime rejects fp32 MemRef states). convert.py
# writes its string name into qwen3_asr.json ("kv_dtype") so the engine picks the
# matching cache dtype.
KV_CACHE_DTYPE = torch.float16
KV_CACHE_DTYPE_STR = "float16" if KV_CACHE_DTYPE == torch.float16 else "float32"


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    v = x.float()
    v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    return (v * weight.float()).to(x.dtype)


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


# ============================ Qwen3 decoder (SALM) ===========================

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
        self.model = hf_model.model          # text model (embed_tokens, layers, norm)
        self.lm_head = hf_model.lm_head
        theta = cfg.rope_theta
        inv = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(MAX_SEQ).float()
        freqs = torch.outer(t, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("rope_cos", emb.cos())
        self.register_buffer("rope_sin", emb.sin())

    def attn_qkv(self, layer, x, cos, sin):
        a = layer.self_attn
        T = x.shape[1]
        q = a.q_proj(x).view(1, T, self.n_heads, self.head_dim)
        k = a.k_proj(x).view(1, T, self.n_kv, self.head_dim)
        v = a.v_proj(x).view(1, T, self.n_kv, self.head_dim)
        q = _rms_norm(q, a.q_norm.weight, self.eps).transpose(1, 2)
        k = _rms_norm(k, a.k_norm.weight, self.eps).transpose(1, 2)
        v = v.transpose(1, 2)
        q = _rope(q, cos, sin)
        k = _rope(k, cos, sin)
        return q, k, v

    def attend(self, layer, q, k, v, mask):
        rep = self.n_heads // self.n_kv
        T = q.shape[2]
        kr = k.repeat_interleave(rep, dim=1)
        vr = v.repeat_interleave(rep, dim=1)
        scores = torch.matmul(q, kr.transpose(-1, -2)) / (self.head_dim ** 0.5)
        scores = scores + mask
        probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        out = torch.matmul(probs, vr)
        out = out.transpose(1, 2).reshape(1, T, self.n_heads * self.head_dim)
        return layer.self_attn.o_proj(out.float())

    def mlp(self, layer, x):
        m = layer.mlp
        return m.down_proj(nn.functional.silu(m.gate_proj(x)) * m.up_proj(x))


class QwenEmbedForCoreAI(nn.Module):
    """ids [1,N] i32 -> embeds [1,N,hidden]."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.embed = hf_model.model.embed_tokens

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed(ids.to(torch.int64))


class QwenStepForCoreAI(nn.Module):
    """One AR step: token at position pos; one-hot cache write; mask idx<=pos."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.core = _QwenCore(hf_model)
        self.embed = hf_model.model.embed_tokens
        c = self.core
        shape = (c.n_layers, 1, c.n_kv, MAX_SEQ, c.head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=KV_CACHE_DTYPE))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=KV_CACHE_DTYPE))
        self.register_buffer("arangeS", torch.arange(MAX_SEQ, dtype=torch.int32))

    def forward(self, tok: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        c = self.core
        p = pos.to(torch.int64).reshape(1)
        x = self.embed(tok.to(torch.int64)).view(1, 1, c.hidden)
        cos = torch.nn.functional.embedding(p, c.rope_cos)
        sin = torch.nn.functional.embedding(p, c.rope_sin)
        onehot = (self.arangeS == pos.reshape(1)).to(KV_CACHE_DTYPE).view(1, 1, MAX_SEQ, 1)
        inv = (1 - onehot)[0]
        keymask = torch.where(
            self.arangeS.view(1, 1, 1, MAX_SEQ) <= pos.reshape(1, 1, 1, 1),
            torch.zeros(1, dtype=x.dtype),
            torch.full((1,), torch.finfo(torch.float32).min, dtype=x.dtype),
        )
        for li, layer in enumerate(c.model.layers):
            h = _rms_norm(x, layer.input_layernorm.weight, c.eps)
            q, k, v = c.attn_qkv(layer, h, cos, sin)
            k = k.to(KV_CACHE_DTYPE)
            v = v.to(KV_CACHE_DTYPE)
            self.k_cache[li] = self.k_cache[li] * inv + (k * onehot)[0]
            self.v_cache[li] = self.v_cache[li] * inv + (v * onehot)[0]
            att = c.attend(layer, q.to(KV_CACHE_DTYPE),
                           self.k_cache[li], self.v_cache[li], keymask)
            x = x + att
            h = _rms_norm(x, layer.post_attention_layernorm.weight, c.eps)
            x = x + c.mlp(layer, h)
        x = _rms_norm(x, c.model.norm.weight, c.eps)
        logits = c.lm_head(x.squeeze(1))
        return torch.argmax(logits, dim=-1).to(torch.int32)


class QwenPrefillCore(nn.Module):
    def __init__(self, hf_model: nn.Module, seq_len: int):
        super().__init__()
        self.core = _QwenCore(hf_model)
        self.seq_len = seq_len
        causal = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min)
        self.register_buffer("causal_mask", torch.triu(causal, diagonal=1).view(1, 1, seq_len, seq_len))


class QwenPrefillForCoreAI(QwenPrefillCore):
    """Caches as PLAIN OUTPUTS (no states): host threads them to the step."""

    def forward(self, embeds, last_idx):
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
        k_out = torch.cat(new_k, dim=0).to(KV_CACHE_DTYPE)
        v_out = torch.cat(new_v, dim=0).to(KV_CACHE_DTYPE)
        return tok, k_out, v_out


# ============================ Qwen3-ASR audio tower ==========================

def audio_out_len(mel_frames: int) -> int:
    """Audio tokens the tower emits for `mel_frames` (must be multiple of 100)."""
    return (mel_frames // 100) * 13


class Qwen3ASRAudioEncoderForCoreAI(nn.Module):
    """Static reimpl of Qwen3ASRAudioEncoder for a FIXED window W (mult of 100).

    mel [1,128,W] + length -> audio_embeds [1, (W//100)*13, output_dim]. Audio is
    pre-padded to exactly W frames (Whisper-style fixed window), so the chunk count
    and the block-diagonal attention mask are static. Uses the real tower's weights.
    """

    def __init__(self, tower: nn.Module, W: int):
        super().__init__()
        self.t = tower
        cfg = tower.config
        assert W % 100 == 0, "window must be a multiple of 100 mel frames"
        self.W = W
        self.nc = W // 100                              # full 100-frame chunks
        S = self.nc * 13                                # audio tokens
        self.S = S
        self.n_mels = cfg.num_mel_bins
        self.d_model = cfg.d_model
        self.nh = cfg.encoder_attention_heads
        self.hd = cfg.d_model // self.nh
        # block-local attention window (after cnn), as the real cu_seqlens builds it
        self.win = 13 * (cfg.n_window_infer // (cfg.n_window * 2))
        blk = torch.arange(S) // self.win
        mask = torch.where(blk[:, None] == blk[None, :], 0.0, float("-inf"))
        self.register_buffer("mask", mask.view(1, 1, S, S))
        self.register_buffer("pos", tower.positional_embedding.positional_embedding[:S].clone())

    def _attn(self, layer, x):
        a = layer.self_attn
        S = x.shape[0]
        q = a.q_proj(x).reshape(S, self.nh, self.hd).transpose(0, 1).unsqueeze(0)
        k = a.k_proj(x).reshape(S, self.nh, self.hd).transpose(0, 1).unsqueeze(0)
        v = a.v_proj(x).reshape(S, self.nh, self.hd).transpose(0, 1).unsqueeze(0)
        s = torch.matmul(q, k.transpose(-1, -2)) * (self.hd ** -0.5) + self.mask
        p = torch.softmax(s, dim=-1)
        o = torch.matmul(p, v).squeeze(0).transpose(0, 1).reshape(S, -1)
        return a.out_proj(o)

    def forward(self, mel: torch.Tensor, mel_length: torch.Tensor) -> torch.Tensor:
        t = self.t
        _ = mel_length  # static window; length is informational
        x = mel.view(1, self.n_mels, self.nc, 100).permute(2, 0, 1, 3).reshape(self.nc, 1, self.n_mels, 100)
        x = F.gelu(t.conv2d1(x)); x = F.gelu(t.conv2d2(x)); x = F.gelu(t.conv2d3(x))
        b, c, f, tt = x.size()
        x = t.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, tt, c * f))   # [nc,13,896]
        x = x.reshape(self.nc * tt, self.d_model)                              # [S,896]
        x = x + self.pos
        for layer in t.layers:
            x = x + self._attn(layer, layer.self_attn_layer_norm(x))
            h = layer.final_layer_norm(x)
            x = x + layer.fc2(layer.activation_fn(layer.fc1(h)))
        x = t.ln_post(x)
        x = t.proj2(t.act(t.proj1(x)))
        return x.unsqueeze(0)                                                   # [1,S,1024]
