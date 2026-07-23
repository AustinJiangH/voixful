"""Stage-by-stage diff: exported .aimodel functions vs the fp32 torch
wrappers. Isolates which graph diverges under the Core AI runtime.

Comparisons use cosine similarity — palette4-vs-fp32 gives ~0.99+; a lowering
bug gives ~0 or NaN.

Usage:
  python -m canary_convert.debug_asset --asset <path> --wav ../../fixtures/sample.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json

import numpy as np
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
from .reference import load_model
from .verify import prompt_ids, split_at_placeholder


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else float("nan")


async def main_async(args) -> int:
    from coreai import runtime as rt

    manifest = json.loads(open(f"{args.asset}/canary.json").read())
    model = load_model()
    tag_id = model.tokenizer.token_to_id(model.audio_locator_tag)
    asr_ids = prompt_ids(model, f"Transcribe the following: {model.audio_locator_tag}")
    pre, post = split_at_placeholder(asr_ids, tag_id)
    mods = extract_modules(model)

    audio, sr = sf.read(args.wav, dtype="float32")
    sig = torch.from_numpy(audio).unsqueeze(0)
    with torch.no_grad():
        mel, mel_len = mods["preprocessor"](input_signal=sig, length=torch.tensor([sig.shape[1]]))

    W = manifest["windows"][0]
    mel_pad = torch.zeros(1, 128, W["mel_frames"])
    mel_pad[:, :, : mel.shape[2]] = mel
    length = int(mel_len[0])

    # ---- torch reference (fp32, unpalettized) ----
    enc_t = CanaryEncoderForCoreAI(mods["encoder"], mods["proj"]).eval()
    with torch.no_grad():
        ref_embeds = enc_t(mel_pad, torch.tensor([length], dtype=torch.int32))
    a_real = (length + 7) // 8

    embed_t = QwenEmbedForCoreAI(mods["llm_asr"]).eval()
    with torch.no_grad():
        pre_e = embed_t(torch.tensor([pre], dtype=torch.int32))
        post_e = embed_t(torch.tensor([post], dtype=torch.int32))
    prompt = torch.cat([pre_e, ref_embeds[:, :a_real], post_e], dim=1)
    P = prompt.shape[1]
    bucket = W["asr_bucket"]

    pf_t = QwenPrefillForCoreAI(mods["llm_asr"], bucket).eval()
    st_t = QwenStepForCoreAI(mods["llm_asr"]).eval()
    st_t.k_cache, st_t.v_cache = pf_t.k_cache, pf_t.v_cache
    padded = torch.zeros(1, bucket, 2048)
    padded[:, :P] = prompt
    with torch.no_grad():
        tok0_t = int(pf_t(padded, torch.tensor([P - 1], dtype=torch.int32)).item())
        tok1_t = int(st_t(torch.tensor([tok0_t], dtype=torch.int32),
                          torch.tensor([P], dtype=torch.int32)).item())
    print(f"torch: first tokens {tok0_t}, {tok1_t}")

    # ---- asset ----
    m = await rt.AIModel.load(args.asset)

    # encode
    enc_fn = m.load_function(f"encode_{W['mel_frames']}")
    out = await enc_fn({
        "mel": rt.NDArray(mel_pad.numpy()),
        "mel_length": rt.NDArray(np.array([length], dtype=np.int32)),
    })
    asset_embeds = out["audio_embeds"].numpy()
    print(f"encode cos={cos(asset_embeds[:, :a_real], ref_embeds[:, :a_real].numpy()):.4f}")

    # embed
    emb_fn = m.load_function(f"embed_{manifest['embed_n']}")
    ids_pad = np.zeros((1, manifest["embed_n"]), dtype=np.int32)
    ids_pad[0, : len(pre)] = pre
    out = await emb_fn({"ids": rt.NDArray(ids_pad)})
    print(f"embed cos={cos(out['embeds'].numpy()[:, :len(pre)], pre_e.numpy()):.4f}")

    # prefill (asset embeds host-assembled exactly like the Swift engine)
    k = np.zeros((28, 1, 8, MAX_SEQ, 128), dtype=np.float32)
    v = np.zeros((28, 1, 8, MAX_SEQ, 128), dtype=np.float32)
    padded_np = padded.numpy().copy()
    pf_fn = m.load_function(f"prefill_asr_{bucket}")
    out = await pf_fn(
        {"embeds": rt.NDArray(padded_np),
         "last_idx": rt.NDArray(np.array([P - 1], dtype=np.int32))},
        state={"k_cache": rt.NDArray(k), "v_cache": rt.NDArray(v)},
    )
    tok0_a = int(out["token"].numpy().ravel()[0])
    print(f"prefill token: asset={tok0_a} torch={tok0_t} "
          f"cacheK cos={cos(k[:, :, :, :P], pf_t.k_cache[:, :, :, :P].detach().numpy()):.4f}")

    # step
    st_fn = m.load_function("step_asr")
    out = await st_fn(
        {"tok": rt.NDArray(np.array([tok0_t], dtype=np.int32)),
         "pos": rt.NDArray(np.array([P], dtype=np.int32))},
        state={"k_cache": rt.NDArray(k), "v_cache": rt.NDArray(v)},
    )
    tok1_a = int(out["token"].numpy().ravel()[0])
    print(f"step token: asset={tok1_a} torch={tok1_t}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="canary-debug-asset")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--wav", required=True)
    args = parser.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())


async def probe2(args) -> int:
    """Hypothesis tests: (a) does torch predict the asset's wrong token when
    reading the LAST bucket row instead of last_idx? (b) encoder cosine by
    row region; (c) detokenize the suspect tokens."""
    from coreai import runtime as rt

    manifest = json.loads(open(f"{args.asset}/canary.json").read())
    model = load_model()
    tag_id = model.tokenizer.token_to_id(model.audio_locator_tag)
    asr_ids = prompt_ids(model, f"Transcribe the following: {model.audio_locator_tag}")
    pre, post = split_at_placeholder(asr_ids, tag_id)
    print("token 198:", repr(model.tokenizer.ids_to_text(torch.tensor([198]))),
          "token 785:", repr(model.tokenizer.ids_to_text(torch.tensor([785]))))
    mods = extract_modules(model)

    audio, sr = sf.read(args.wav, dtype="float32")
    sig = torch.from_numpy(audio).unsqueeze(0)
    with torch.no_grad():
        mel, mel_len = mods["preprocessor"](input_signal=sig, length=torch.tensor([sig.shape[1]]))
    W = manifest["windows"][0]
    mel_pad = torch.zeros(1, 128, W["mel_frames"])
    mel_pad[:, :, : mel.shape[2]] = mel
    length = int(mel_len[0])

    enc_t = CanaryEncoderForCoreAI(mods["encoder"], mods["proj"]).eval()
    with torch.no_grad():
        ref_embeds = enc_t(mel_pad, torch.tensor([length], dtype=torch.int32))
    a_real = (length + 7) // 8

    m = await rt.AIModel.load(args.asset)
    enc_fn = m.load_function(f"encode_{W['mel_frames']}")
    out = await enc_fn({
        "mel": rt.NDArray(mel_pad.numpy()),
        "mel_length": rt.NDArray(np.array([length], dtype=np.int32)),
    })
    ae = out["audio_embeds"].numpy()
    for lo in range(0, a_real, 16):
        hi = min(lo + 16, a_real)
        print(f"  enc rows {lo}:{hi} cos={cos(ae[:, lo:hi], ref_embeds[:, lo:hi].numpy()):.4f}")

    embed_t = QwenEmbedForCoreAI(mods["llm_asr"]).eval()
    with torch.no_grad():
        pre_e = embed_t(torch.tensor([pre], dtype=torch.int32))
        post_e = embed_t(torch.tensor([post], dtype=torch.int32))
    prompt = torch.cat([pre_e, ref_embeds[:, :a_real], post_e], dim=1)
    P = prompt.shape[1]
    bucket = W["asr_bucket"]
    padded = torch.zeros(1, bucket, 2048)
    padded[:, :P] = prompt
    pf_t = QwenPrefillForCoreAI(mods["llm_asr"], bucket).eval()
    with torch.no_grad():
        for idx in (P - 1, bucket - 1, 0):
            t = int(pf_t(padded, torch.tensor([idx], dtype=torch.int32)).item())
            print(f"torch prefill last_idx={idx} -> token {t}")
    return 0


def main2(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True)
    parser.add_argument("--wav", required=True)
    args = parser.parse_args(argv)
    return asyncio.run(probe2(args))
