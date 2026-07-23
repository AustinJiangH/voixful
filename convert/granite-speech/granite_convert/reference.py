"""Reference (ground-truth) transcription via the HF trust_remote_code model.

Run this FIRST: it downloads the checkpoint and prints the transcript the Core
AI port must reproduce. Keep the printed transcript next to the asset — the
Swift engine's parity gate compares against it.

Usage:
  python -m granite_convert.reference --wav ../../fixtures/sample.wav
"""

from __future__ import annotations

import argparse
import time

import soundfile as sf
import torch

MODEL_ID = "ibm-granite/granite-speech-4.1-2b-nar"


def load_model(dtype=torch.float32):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=dtype, attn_implementation="sdpa"
    ).eval()
    return processor, model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="granite-reference")
    parser.add_argument("--wav", required=True)
    args = parser.parse_args(argv)

    audio, sr = sf.read(args.wav, dtype="float32")
    assert sr == 16000, f"expected 16 kHz, got {sr}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    processor, model = load_model()
    inputs = processor([torch.from_numpy(audio)], return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        out = model.transcribe(**inputs)
    dt = time.time() - t0
    text = processor.batch_decode(out.preds)
    enc_text = processor.batch_decode(out.encoder_preds)
    print(f"[granite/reference] {len(audio)/sr:.1f}s audio, {dt:.1f}s wall (fp32 cpu)")
    print(f"encoder-CTC hypothesis: {enc_text[0]!r}")
    print(f"final transcript:       {text[0]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
