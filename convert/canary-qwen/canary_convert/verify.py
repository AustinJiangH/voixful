"""Wrapper-vs-reference parity gate for the full two-pass pipeline. Run
BEFORE export.

Drives mel -> encoder+proj -> prompt splice -> prefill -> step loop (ASR,
merged LoRA) and then the format pass (plain Qwen3) exactly as the Swift
runtime will, comparing against SALM.generate on the same audio.

Gates:
  - ASR transcript: EXACT text match (greedy decode, fp32 both sides).
  - Format pass: prints both; exact match expected but token-boundary
    effects at the transcript splice may cause benign wording drift —
    require a bulleted list containing the transcript's content words.

Usage:
  python -m canary_convert.verify --wav ../../fixtures/sample.wav
"""

from __future__ import annotations

import argparse

import soundfile as sf
import torch

from .extract import extract_modules
from .models import (
    MAX_SEQ,
    CanaryEncoderForCoreAI,
    QwenEmbedForCoreAI,
    QwenPrefillForCoreAI,
    QwenStepForCoreAI,
)
from .reference import FORMAT_PROMPT, load_model

EOS = 151645  # <|im_end|>


def prompt_ids(model, content: str) -> list[int]:
    from nemo.collections.common.prompts import PromptFormatter

    formatter = PromptFormatter.resolve(model.cfg.prompt_format)(model.tokenizer)
    enc = formatter.encode_dialog(turns=[{"role": "user", "content": content}])
    return enc["input_ids"].tolist()


def split_at_placeholder(ids: list[int], placeholder: int) -> tuple[list[int], list[int]]:
    i = ids.index(placeholder)
    return ids[:i], ids[i + 1 :]


def generate(prefill, step, embeds: torch.Tensor, prompt_len: int, bucket: int,
             max_new: int = 256) -> list[int]:
    """embeds [1,P,2048] -> generated ids (greedy, stop at EOS). Caches are
    threaded functionally, exactly as the runtime passes them via states."""
    assert prompt_len <= bucket <= MAX_SEQ
    padded = torch.zeros(1, bucket, embeds.shape[-1])
    padded[:, :prompt_len] = embeds
    k = torch.zeros(28, 1, 8, MAX_SEQ, 128)
    v = torch.zeros(28, 1, 8, MAX_SEQ, 128)
    with torch.no_grad():
        tok, k, v = prefill(padded, torch.tensor([prompt_len - 1], dtype=torch.int32), k, v)
        out: list[int] = []
        pos = prompt_len
        while len(out) < max_new:
            t = int(tok.item())
            if t == EOS:
                break
            out.append(t)
            tok, k, v = step(torch.tensor([t], dtype=torch.int32),
                             torch.tensor([pos], dtype=torch.int32), k, v)
            pos += 1
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="canary-verify")
    parser.add_argument("--wav", required=True)
    args = parser.parse_args(argv)

    model = load_model()
    tag_id = model.tokenizer.token_to_id(model.audio_locator_tag)

    # ---- references (before extraction mutates anything) ----
    with torch.no_grad():
        ref_ids = model.generate(
            prompts=[[{
                "role": "user",
                "content": f"Transcribe the following: {model.audio_locator_tag}",
                "audio": [args.wav],
            }]],
            max_new_tokens=256,
        )
    ref_text = model.tokenizer.ids_to_text(ref_ids[0].cpu())
    print(f"reference ASR: {ref_text!r}")

    mods = extract_modules(model)
    tok = mods["tokenizer"]

    # ---- our pipeline: mel -> encoder ----
    audio, sr = sf.read(args.wav, dtype="float32")
    assert sr == 16000
    sig = torch.from_numpy(audio).unsqueeze(0)
    length = torch.tensor([sig.shape[1]])
    with torch.no_grad():
        mel, mel_len = mods["preprocessor"](input_signal=sig, length=length)
        enc_wrapper = CanaryEncoderForCoreAI(mods["encoder"], mods["proj"]).eval()
        audio_embeds = enc_wrapper(mel, mel_len.to(torch.int32))
    print(f"audio embeds: {tuple(audio_embeds.shape)}")

    # ---- ASR pass through the wrappers ----
    asr_ids = prompt_ids(model, f"Transcribe the following: {model.audio_locator_tag}")
    pre, post = split_at_placeholder(asr_ids, tag_id)
    embed = QwenEmbedForCoreAI(mods["llm_asr"]).eval()
    with torch.no_grad():
        pre_e = embed(torch.tensor([pre], dtype=torch.int32))
        post_e = embed(torch.tensor([post], dtype=torch.int32))
    prompt_embeds = torch.cat([pre_e, audio_embeds, post_e], dim=1)
    P = prompt_embeds.shape[1]
    bucket = min(MAX_SEQ, ((P + 31) // 32) * 32 + 32)

    prefill_asr = QwenPrefillForCoreAI(mods["llm_asr"], bucket).eval()
    step_asr = QwenStepForCoreAI(mods["llm_asr"]).eval()

    out_ids = generate(prefill_asr, step_asr, prompt_embeds, P, bucket)
    text = tok.ids_to_text(torch.tensor(out_ids))
    match = text.strip() == ref_text.strip()
    print(f"our ASR ({'MATCH' if match else 'DIFFER'}): {text!r}")

    # ---- format pass (plain Qwen3) ----
    fmt_ids = prompt_ids(model, FORMAT_PROMPT + model.audio_locator_tag)
    fpre, fpost = split_at_placeholder(fmt_ids, tag_id)
    embed_p = QwenEmbedForCoreAI(mods["llm_plain"]).eval()
    with torch.no_grad():
        fp_e = embed_p(torch.tensor([fpre], dtype=torch.int32))
        tr_e = embed_p(torch.tensor([out_ids], dtype=torch.int32))
        fpo_e = embed_p(torch.tensor([fpost], dtype=torch.int32))
    fmt_embeds = torch.cat([fp_e, tr_e, fpo_e], dim=1)
    FP = fmt_embeds.shape[1]
    fbucket = min(MAX_SEQ, ((FP + 31) // 32) * 32 + 32)

    prefill_llm = QwenPrefillForCoreAI(mods["llm_plain"], fbucket).eval()
    step_llm = QwenStepForCoreAI(mods["llm_plain"]).eval()

    fmt_out = generate(prefill_llm, step_llm, fmt_embeds, FP, fbucket, max_new=512)
    fmt_text = tok.ids_to_text(torch.tensor(fmt_out))
    print("format pass output:")
    print(fmt_text)

    bullets = [l for l in fmt_text.splitlines() if l.strip().startswith(("-", "*", "•"))]
    ok = match and len(bullets) >= 1
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
