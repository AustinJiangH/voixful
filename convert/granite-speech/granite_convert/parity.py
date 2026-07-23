"""Padding-invariance probe — gates the whole static-window export design.

The Core AI export uses fixed encoder windows (multiples of the 200-frame
attention block). Granite's encoder does NOT mask padded frames internally
(only the partial last block is masked), so window padding is only safe if
padding the AUDIO with silence — in-distribution for the model — leaves the
transcript unchanged with an all-ones attention mask.

This script compares, per window size:
  A) reference: transcribe(exact-length audio)
  B) padded:    transcribe(audio zero-padded to the window, mask all ones)

If the texts match across windows and clips, the static design is sound.

Usage:
  python -m granite_convert.parity --wav ../../fixtures/sample.wav
"""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf
import torch

from .reference import load_model

# Encoder frames are 50 Hz; windows must be multiples of the 200-frame
# (4 s) attention block. samples = frames * 320.
WINDOW_FRAMES = [200, 400, 800, 1600]


def transcribe(processor, model, audio: np.ndarray) -> tuple[str, str]:
    inputs = processor([torch.from_numpy(audio)], return_tensors="pt")
    with torch.no_grad():
        out = model.transcribe(**inputs)
    return (
        processor.batch_decode(out.preds)[0],
        processor.batch_decode(out.encoder_preds)[0],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="granite-parity")
    parser.add_argument("--wav", required=True)
    args = parser.parse_args(argv)

    audio, sr = sf.read(args.wav, dtype="float32")
    assert sr == 16000
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    processor, model = load_model()
    ref_text, ref_enc = transcribe(processor, model, audio)
    print(f"reference ({len(audio)/sr:.1f}s): {ref_text!r}")
    print(f"  encoder hypothesis:            {ref_enc!r}")

    ok = True
    for frames in WINDOW_FRAMES:
        samples = frames * 320
        if samples < len(audio):
            continue
        padded = np.zeros(samples, dtype=np.float32)
        padded[: len(audio)] = audio
        text, enc = transcribe(processor, model, padded)
        match = "MATCH" if text == ref_text else "DIFFER"
        enc_match = "MATCH" if enc == ref_enc else "DIFFER"
        print(f"window {frames:5d} ({frames/50:5.1f}s): final={match} encoder-ctc={enc_match}")
        if text != ref_text:
            ok = False
            print(f"  padded: {text!r}")
    print("PADDING-INVARIANT" if ok else "PADDING CHANGES OUTPUT — need masked export design")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
