"""One-shot file transcription demo.

Transcribes a WAV file through the cached NeMo model and prints the result.
Useful for: confirming the pipeline works without granting mic access; running
a smoke test on a known-good sample.

Run:
    python -m nemotron_convert.file_demo path/to/audio.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="file-demo",
        description="Transcribe a single audio file via NeMo and print the result.",
    )
    parser.add_argument("audio", type=Path, help="Path to a WAV / FLAC / OGG file.")
    parser.add_argument(
        "--model-id",
        default="nvidia/nemotron-speech-streaming-en-0.6b",
        help="HuggingFace checkpoint id (must already be cached).",
    )
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"[file-demo] {args.audio} does not exist", file=sys.stderr)
        return 2

    logging.getLogger("nemo_logger").setLevel(logging.WARNING)

    print(f"[file-demo] Loading {args.model_id} …", flush=True)
    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(args.model_id)
    model.eval()

    audio, sample_rate = sf.read(str(args.audio))
    if audio.ndim > 1:
        # Down-mix multichannel to mono by averaging.
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    if sample_rate != 16000:
        print(f"[file-demo] resampling {sample_rate} -> 16000 …", flush=True)
        # NeMo expects 16 kHz mono Float32. Use scipy for resampling.
        from scipy.signal import resample_poly
        # GCD-based ratio for speed
        from math import gcd
        g = gcd(sample_rate, 16000)
        up, down = 16000 // g, sample_rate // g
        audio = resample_poly(audio, up, down).astype(np.float32)
        sample_rate = 16000

    duration = len(audio) / sample_rate
    print(f"[file-demo] {args.audio.name}: {duration:.2f}s @ {sample_rate} Hz", flush=True)

    result = model.transcribe([audio], batch_size=1, verbose=False)
    r0 = result[0]
    text = getattr(r0, "text", None) or (r0 if isinstance(r0, str) else str(r0))
    print()
    print(text)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
