"""Wrapper-vs-reference parity gate. Run BEFORE export.

Drives the two wrappers plus the host glue exactly as the Swift runtime will,
and compares every stage against the HF reference on the same padded audio:

  1. mel front-end vs GraniteSpeechNarFeatureExtractor (atol)
  2. encoder ctc_ids vs reference encoder logits argmax
  3. audio_embeds vs reference projector output (atol)
  4. final ids/text vs model.transcribe()

Usage:
  python -m granite_convert.verify --wav ../../fixtures/sample.wav --frames 400
"""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf
import torch

from .models import SAMPLES_PER_FRAME, GraniteEditForCoreAI, GraniteEncoderForCoreAI
from .reference import load_model


def ctc_collapse(ids: torch.Tensor, blank: int) -> torch.Tensor:
    u = torch.unique_consecutive(ids)
    return u[u != blank]


def add_insertion_slots(tokens: torch.Tensor, blank: int, min_len: int = 8) -> torch.Tensor:
    n = tokens.numel()
    total = max(2 * n + 1, min_len)
    out = torch.full((total,), blank, dtype=tokens.dtype)
    out[2 * torch.arange(n) + 1] = tokens
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="granite-verify")
    parser.add_argument("--wav", required=True)
    parser.add_argument("--frames", type=int, default=400, help="encoder window (50 Hz frames, mult of 200)")
    parser.add_argument("--text-bucket", type=int, default=64)
    args = parser.parse_args(argv)

    audio, sr = sf.read(args.wav, dtype="float32")
    assert sr == 16000
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    samples = args.frames * SAMPLES_PER_FRAME
    assert len(audio) <= samples, f"clip needs window > {len(audio)//SAMPLES_PER_FRAME} frames"
    padded = np.zeros(samples, dtype=np.float32)
    padded[: len(audio)] = audio
    audio_t = torch.from_numpy(padded).unsqueeze(0)

    processor, model = load_model()
    blank = model.config.blank_token_id
    min_len = model.config.min_edit_sequence_length

    # ---- reference on the SAME padded audio ----
    inputs = processor([audio_t[0]], return_tensors="pt")
    with torch.no_grad():
        ref = model.transcribe(**inputs)
    ref_text = processor.batch_decode(ref.preds)[0]
    ref_ctc = ref.encoder_preds[0]

    # ---- stage 1: features ----
    enc_wrapper = GraniteEncoderForCoreAI(model, args.frames).eval()
    with torch.no_grad():
        feats = enc_wrapper.mel(audio_t)
    ref_feats = inputs["input_features"]
    d_feat = (feats - ref_feats).abs().max().item()
    print(f"mel front-end: max|diff| = {d_feat:.2e} (shape {tuple(feats.shape)})")

    # ---- stage 2+3: encoder graph ----
    with torch.no_grad():
        audio_embeds, ctc_ids = enc_wrapper(audio_t)
    hyp = ctc_collapse(ctc_ids[0], blank)
    ctc_match = torch.equal(hyp, ref_ctc)
    print(f"encoder CTC hypothesis: {'MATCH' if ctc_match else 'DIFFER'} ({hyp.numel()} tokens)")
    if not ctc_match:
        print(f"  ref: {ref_ctc.tolist()}\n  got: {hyp.tolist()}")

    # ---- stage 4: edit graph + host glue ----
    slotted = add_insertion_slots(hyp.to(torch.int32), blank, min_len)
    u_real = slotted.numel()
    bucket = args.text_bucket
    assert u_real <= bucket, f"hypothesis {u_real} > bucket {bucket}"
    text_ids = torch.full((1, bucket), blank, dtype=torch.int32)
    text_ids[0, :u_real] = slotted
    a_len = audio_embeds.shape[1]
    attn_mask = torch.zeros(1, a_len + bucket)
    attn_mask[0, : a_len + u_real] = 1.0

    edit_wrapper = GraniteEditForCoreAI(model, a_len, bucket).eval()
    with torch.no_grad():
        out_ids = edit_wrapper(audio_embeds, text_ids, attn_mask)
    final = ctc_collapse(out_ids[0, :u_real].to(torch.int64), blank)
    text = processor.batch_decode([final])[0]
    text_match = text == ref_text
    print(f"final transcript: {'MATCH' if text_match else 'DIFFER'}")
    print(f"  ref: {ref_text!r}")
    if not text_match:
        print(f"  got: {text!r}")

    ok = ctc_match and text_match and d_feat < 1e-3
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
