"""Pull export-ready modules out of the SALM checkpoint.

Returns fp32 CPU modules:
  encoder      — NeMo ConformerEncoder (32L, d1024)
  proj         — Linear 1024 -> 2048
  preprocessor — NeMo mel front-end (host-parity reference)
  llm_asr      — Qwen3ForCausalLM with the ASR LoRA MERGED (transcription)
  llm_plain    — pristine Qwen3-1.7B-instruct (adapters removed, unmerged)
  tokenizer    — SALM's AutoTokenizer (Qwen3)
"""

from __future__ import annotations

import copy
from typing import Any

import torch


def extract_modules(model) -> dict[str, Any]:
    model = model.float().eval()

    llm_asr = copy.deepcopy(model.llm).merge_and_unload().eval()
    llm_plain = copy.deepcopy(model.llm).unload().eval()

    # SALM detaches the embedding from the LLM (it embeds prompt tokens itself
    # so audio frames can be spliced in); reattach for standalone use. The
    # table is frozen in SALM training, so one shared copy is exact for both.
    embed = copy.deepcopy(model.embed_tokens)
    llm_asr.model.embed_tokens = embed
    llm_plain.model.embed_tokens = embed

    # Sanity: LoRA touched only q/v — everything else must be identical, and
    # merged q_proj must differ from plain.
    sd_a, sd_p = llm_asr.state_dict(), llm_plain.state_dict()
    assert sd_a.keys() == sd_p.keys()
    q_key = "model.layers.0.self_attn.q_proj.weight"
    up_key = "model.layers.0.mlp.up_proj.weight"
    assert not torch.equal(sd_a[q_key], sd_p[q_key]), "LoRA merge did not change q_proj"
    assert torch.equal(sd_a[up_key], sd_p[up_key]), "non-LoRA weights diverged"

    return {
        "encoder": model.perception.encoder,
        "proj": model.perception.proj,
        "preprocessor": model.perception.preprocessor,
        "llm_asr": llm_asr,
        "llm_plain": llm_plain,
        "tokenizer": model.tokenizer,
        "audio_locator_tag": model.audio_locator_tag,
    }
