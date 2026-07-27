"""Pull export-ready modules out of the Qwen3-ASR checkpoint.

Returns fp32 CPU modules:
  audio_tower  — the real Qwen3ASRAudioEncoder (wrapped statically for export)
  text_lm      — a CausalLM adapter over the Qwen3 text decoder exposing
                 `.model` / `.lm_head` / `.config` (what the SALM decode wrappers
                 read) plus a `forward(ids)` for palettization calibration
  tokenizer    — the Qwen2Tokenizer
  pre_ids/post_ids — the ASR prompt token ids around the audio placeholder run
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .reference import AUDIO_PAD


class TextCausalLM(nn.Module):
    """CausalLM view of the Qwen3-ASR thinker text decoder.

    Exposes `.model` (Qwen3ASRThinkerTextModel), `.lm_head`, `.config` so the
    Canary SALM wrappers (_QwenCore/QwenStep/QwenPrefill/QwenEmbed) work
    unchanged, and a `forward(input_ids)->logits` so coreai_opt palettization can
    calibrate the whole decoder in one pass.
    """

    def __init__(self, thinker: nn.Module):
        super().__init__()
        self.model = thinker.model
        self.lm_head = thinker.lm_head
        self.config = thinker.model.config

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids.to(torch.int64))
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return self.lm_head(h)


def build_asr_prompt_ids(processor) -> tuple[list[int], list[int]]:
    """The constant pre/post token ids around the audio placeholder run.

    Builds the ASR chat prompt and splits at the <|audio_pad|> run: everything
    before it is `pre_ids` (…<|audio_start|>), everything after is `post_ids`
    (<|audio_end|><|im_end|>… assistant). The number of audio pads varies with
    clip length but the pre/post are constant, so a 1-pad probe suffices.
    """
    tok = processor.tokenizer
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "audio", "audio": "x.wav"},
                                       {"type": "text", "text": ""}]}],
        add_generation_prompt=True, tokenize=False)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    # The template renders exactly one <|audio_pad|>; split around it.
    i = ids.index(AUDIO_PAD)
    j = len(ids) - 1 - ids[::-1].index(AUDIO_PAD)
    return ids[:i], ids[j + 1:]


def extract_modules(model, processor) -> dict[str, Any]:
    model = model.float().eval()
    thinker = model.thinker
    pre_ids, post_ids = build_asr_prompt_ids(processor)
    return {
        "audio_tower": thinker.audio_tower.eval(),
        "text_lm": TextCausalLM(thinker).eval(),
        "tokenizer": processor.tokenizer,
        "processor": processor,
        "pre_ids": pre_ids,
        "post_ids": post_ids,
    }
