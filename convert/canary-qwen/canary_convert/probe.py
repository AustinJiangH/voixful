"""Print the SALM module structure + configs needed to plan the export.

Usage:
  python -m canary_convert.probe
"""

from __future__ import annotations

import torch

from .reference import load_model


def main() -> int:
    model = load_model()
    print("=== top-level attrs ===")
    print("audio_locator_tag:", model.audio_locator_tag)
    print("perception:", type(model.perception).__name__)
    print("  preprocessor:", type(model.perception.preprocessor).__name__)
    print("  encoder:", type(model.perception.encoder).__name__)
    enc = model.perception.encoder
    print("  encoder d_model/layers:", getattr(enc, "d_model", "?"), len(enc.layers))
    print("  modality_adapter:", type(model.perception.modality_adapter).__name__)
    for name, mod in model.perception.named_children():
        if name not in ("preprocessor", "encoder", "modality_adapter"):
            print("  extra child:", name, type(mod).__name__)
    print("llm:", type(model.llm).__name__)
    inner = model.llm
    print("llm config:", inner.config if hasattr(inner, "config") else "?")
    print("=== projection search ===")
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "perception" in name:
            print(" ", name, mod.in_features, "->", mod.out_features)
    print("=== tokenizer ===")
    tok = model.tokenizer
    print("tokenizer type:", type(tok).__name__)
    print("vocab size:", tok.vocab_size if hasattr(tok, "vocab_size") else "?")
    print("=== prompt formatter probe ===")
    ids = model.tokenizer.text_to_ids("Transcribe the following: ")
    print("'Transcribe the following: ' ->", ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
