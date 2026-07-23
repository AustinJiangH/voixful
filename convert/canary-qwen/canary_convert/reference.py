"""Reference (ground-truth) runs via NeMo SALM. Run FIRST.

Prints, for the given wav:
  1. ASR-mode transcript (LoRA active): the exact text the Core AI port must
     reproduce.
  2. LLM-mode formatted output (adapter disabled = vanilla Qwen3-1.7B):
     the transcript reformatted as bullets — the format-pass ground truth.

Qwen3 is a thinking model; the format prompt appends "/no_think" (Qwen3's
soft switch) so the output is the formatted text directly.

Usage:
  python -m canary_convert.reference --wav ../../fixtures/sample.wav
"""

from __future__ import annotations

import argparse
import time

import torch

MODEL_ID = "nvidia/canary-qwen-2.5b"

FORMAT_PROMPT = (
    "Reformat the following transcript as a markdown bulleted list of the "
    "key points. Keep the speaker's wording; do not summarize away details. "
    "Output only the list. /no_think\n\n"
)


def load_model():
    from nemo.collections.speechlm2.models import SALM

    model = SALM.from_pretrained(MODEL_ID)
    model = model.eval()
    return model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="canary-reference")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--skip-format", action="store_true")
    args = parser.parse_args(argv)

    model = load_model()

    t0 = time.time()
    with torch.no_grad():
        answer_ids = model.generate(
            prompts=[[{
                "role": "user",
                "content": f"Transcribe the following: {model.audio_locator_tag}",
                "audio": [args.wav],
            }]],
            max_new_tokens=256,
        )
    transcript = model.tokenizer.ids_to_text(answer_ids[0].cpu())
    print(f"[canary/reference] ASR mode ({time.time() - t0:.1f}s):")
    print(f"  {transcript!r}")

    if not args.skip_format:
        t1 = time.time()
        with torch.no_grad(), model.llm.disable_adapter():
            fmt_ids = model.generate(
                prompts=[[{
                    "role": "user",
                    "content": FORMAT_PROMPT + transcript,
                }]],
                max_new_tokens=512,
            )
        formatted = model.tokenizer.ids_to_text(fmt_ids[0].cpu())
        print(f"[canary/reference] LLM mode format ({time.time() - t1:.1f}s):")
        print(formatted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
