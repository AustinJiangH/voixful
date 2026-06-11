"""Live microphone transcription demo.

Captures audio from the default input device at 16 kHz mono, maintains a
rolling N-second buffer, and runs NeMo's transcribe on it once per HOP
seconds. New text is printed in place (overwriting the prior line) so the
caption appears to update as you speak.

Uses the in-process NeMo model, NOT the Core AI `.aimodel` artifact. This
is the fastest path to a working live-caption demo while the Core AI Swift
runtime catches up. The end-to-end Swift+CoreAI demo (target architecture
per PLAN.md) is in `Sources/NemotronASRCLI/main.swift` mic subcommand and
`Demos/NemotronDemo/` — both load the same checkpoint via the converted
.aimodel.

Run:
    python -m nemotron_convert.mic_demo
"""

from __future__ import annotations

import argparse
import queue
import signal
import sys
import threading
import time

import numpy as np
import sounddevice as sd


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="mic-demo",
        description="Live microphone transcription via NeMo. Speak; transcript updates ~once per hop.",
    )
    parser.add_argument(
        "--model-id",
        default="nvidia/nemotron-speech-streaming-en-0.6b",
        help="HuggingFace checkpoint id (must already be cached).",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=4.0,
        help="Rolling buffer length the model sees on each transcribe call.",
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=0.8,
        help="How often to run transcribe (smaller = lower latency, more CPU).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="sounddevice input device index (default: system default).",
    )
    args = parser.parse_args()

    print(f"[mic-demo] Loading {args.model_id} …", flush=True)
    # Suppress NeMo's chatty info logs during inference.
    import logging
    logging.getLogger("nemo_logger").setLevel(logging.WARNING)

    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(args.model_id)
    model.eval()

    sample_rate = 16000
    window_samples = int(sample_rate * args.window_seconds)
    buffer = np.zeros(window_samples, dtype=np.float32)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()

    def audio_callback(indata, _frames, _time, status):
        if status:
            # Don't crash the audio thread; report once.
            print(f"\n[mic-demo] audio status: {status}", file=sys.stderr)
        # indata is shape (frames, channels); we requested channels=1, so flatten.
        audio_queue.put(indata.copy())

    def handle_sigint(_signum, _frame):
        stop_event.set()
    signal.signal(signal.SIGINT, handle_sigint)

    print("[mic-demo] Ready. Speak into the mic. Ctrl+C to stop.")
    print("[mic-demo] (Transcript line updates in place ↓)")
    print()

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        device=args.device,
        blocksize=int(sample_rate * 0.1),  # 100 ms blocks
    )

    last_text = ""
    with stream:
        last_run = time.monotonic()
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.1).flatten()
            except queue.Empty:
                continue

            # Slide the rolling window: drop the oldest len(chunk) samples,
            # append the new chunk.
            if len(chunk) >= window_samples:
                buffer = chunk[-window_samples:]
            else:
                buffer = np.concatenate([buffer[len(chunk):], chunk])

            now = time.monotonic()
            if now - last_run < args.hop_seconds:
                continue
            last_run = now

            # Audio level (RMS) — gate on a low threshold so we don't transcribe pure silence.
            rms = float(np.sqrt(np.mean(buffer ** 2)))
            if rms < 0.005:
                continue

            try:
                result = model.transcribe(
                    [buffer.copy()],
                    batch_size=1,
                    verbose=False,
                )
            except Exception as exc:
                print(f"\n[mic-demo] transcribe failed: {exc}", file=sys.stderr)
                continue

            # NeMo's transcribe returns either [Hypothesis] or [str] depending on
            # version. Handle both.
            r0 = result[0]
            text = getattr(r0, "text", None) or (r0 if isinstance(r0, str) else str(r0))
            text = text.strip()

            if text and text != last_text:
                # Overwrite the prior line; pad to clear residual chars.
                pad = " " * max(0, len(last_text) - len(text))
                print(f"\r{text}{pad}", end="", flush=True)
                last_text = text

    print("\n[mic-demo] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
