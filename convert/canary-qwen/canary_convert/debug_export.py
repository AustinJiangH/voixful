"""Export ONLY prefill_asr_192 + step_asr (fp32, no palettize) to a scratch
asset for fast lowering-fix iteration; then token-compare against torch.

Usage:
  python -m canary_convert.debug_export --out /tmp/canary-debug.aimodel
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import coreai_torch

from .export import export_prefill, export_step
from .extract import extract_modules
from .models import QwenPrefillForCoreAI, QwenStepForCoreAI
from .reference import load_model

BUCKET = 192


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    model = load_model()
    mods = extract_modules(model)
    llm = mods["llm_asr"]

    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        export_prefill(QwenPrefillForCoreAI(llm, BUCKET).eval()),
        entrypoint_name=f"prefill_asr_{BUCKET}",
        input_names=["embeds", "last_idx"],
        state_names=["k_cache", "v_cache"],
        output_names=["token"],
    )
    print("exported prefill")
    converter.add_exported_program(
        export_step(QwenStepForCoreAI(llm).eval()),
        entrypoint_name="step_asr",
        input_names=["tok", "pos"],
        state_names=["k_cache", "v_cache"],
        output_names=["token"],
    )
    print("exported step")
    prog = converter.to_coreai()
    prog.optimize()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    prog.save_asset(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


async def check_async(args) -> int:
    """Token-compare the scratch asset's prefill/step against torch, then
    greedy-decode 15 tokens through the asset."""
    import json

    import numpy as np
    import soundfile as sf
    import torch

    from coreai import runtime as rt

    from .models import MAX_SEQ, CanaryEncoderForCoreAI, QwenEmbedForCoreAI
    from .verify import prompt_ids, split_at_placeholder

    model = load_model()
    tag_id = model.tokenizer.token_to_id(model.audio_locator_tag)
    asr_ids = prompt_ids(model, f"Transcribe the following: {model.audio_locator_tag}")
    pre, post = split_at_placeholder(asr_ids, tag_id)
    mods = extract_modules(model)

    audio, sr = sf.read(args.wav, dtype="float32")
    sig = torch.from_numpy(audio).unsqueeze(0)
    with torch.no_grad():
        mel, mel_len = mods["preprocessor"](input_signal=sig, length=torch.tensor([sig.shape[1]]))
    mel_pad = torch.zeros(1, 128, 1000)
    mel_pad[:, :, : mel.shape[2]] = mel
    length = int(mel_len[0])
    enc_t = CanaryEncoderForCoreAI(mods["encoder"], mods["proj"]).eval()
    embed_t = QwenEmbedForCoreAI(mods["llm_asr"]).eval()
    with torch.no_grad():
        ref_embeds = enc_t(mel_pad, torch.tensor([length], dtype=torch.int32))
        pre_e = embed_t(torch.tensor([pre], dtype=torch.int32))
        post_e = embed_t(torch.tensor([post], dtype=torch.int32))
    a_real = (length + 7) // 8
    prompt = torch.cat([pre_e, ref_embeds[:, :a_real], post_e], dim=1)
    P = prompt.shape[1]
    padded = torch.zeros(1, BUCKET, 2048)
    padded[:, :P] = prompt

    # Free the ~20 GB of torch models before the runtime maps the fp32
    # asset — holding both segfaults the process (OOM).
    import gc

    padded_np = padded.numpy().copy()
    P_val = P
    del model, mods, enc_t, embed_t, ref_embeds, pre_e, post_e, prompt, padded, mel, mel_pad, sig
    gc.collect()

    m = await rt.AIModel.load(args.out)
    k = rt.NDArray(np.zeros((28, 1, 8, MAX_SEQ, 128), dtype=np.float32))
    v = rt.NDArray(np.zeros((28, 1, 8, MAX_SEQ, 128), dtype=np.float32))
    pf = m.load_function(f"prefill_asr_{BUCKET}")
    out = await pf(
        {"embeds": rt.NDArray(padded_np),
         "last_idx": rt.NDArray(np.array([P_val - 1], dtype=np.int32))},
        state={"k_cache": k, "v_cache": v},
    )
    tok = int(out["token"].numpy().ravel()[0])
    print(f"prefill token: asset={tok} (torch=785 'The')")
    st = m.load_function("step_asr")
    ids = []
    pos = P_val
    while len(ids) < 15 and tok != 151645:
        ids.append(tok)
        out = await st(
            {"tok": rt.NDArray(np.array([tok], dtype=np.int32)),
             "pos": rt.NDArray(np.array([pos], dtype=np.int32))},
            state={"k_cache": k, "v_cache": v},
        )
        tok = int(out["token"].numpy().ravel()[0])
        pos += 1
    print("decoded ids:", ids)
    return 0


def check(argv=None) -> int:
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--wav", required=True)
    args = parser.parse_args(argv)
    return asyncio.run(check_async(args))


def export_minimal(argv=None) -> int:
    """Palette4 minimal asset (prefill_asr_192 + step_asr) with a cached
    palettized LLM so iterations skip the 15-min kmeans. --variant selects
    graph experiments for the specializer-crash bisect."""
    import torch

    import coreai_torch

    from .convert import _palettize
    from .export import export_prefill, export_step
    from .models import QwenPrefillForCoreAI, QwenStepForCoreAI

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", default="/tmp/canary-llm-asr-palette4.pt")
    args = parser.parse_args(argv)

    try:
        llm = torch.load(args.cache, weights_only=False)
        print(f"loaded palettized llm from {args.cache}")
    except Exception:
        model = load_model()
        mods = extract_modules(model)
        llm = _palettize(mods["llm_asr"], (torch.zeros(1, 8, dtype=torch.int64),), "palette4")
        torch.save(llm, args.cache)
        print(f"palettized + cached to {args.cache}")

    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        export_prefill(QwenPrefillForCoreAI(llm, BUCKET).eval()),
        entrypoint_name=f"prefill_asr_{BUCKET}",
        input_names=["embeds", "last_idx"],
        state_names=["k_cache", "v_cache"],
        output_names=["token"],
    )
    print("exported prefill")
    converter.add_exported_program(
        export_step(QwenStepForCoreAI(llm).eval()),
        entrypoint_name="step_asr",
        input_names=["tok", "pos"],
        state_names=["k_cache", "v_cache"],
        output_names=["token"],
    )
    print("exported step")
    prog = converter.to_coreai()
    prog.optimize()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    prog.save_asset(out)
    print(f"wrote {out}")
    return 0
