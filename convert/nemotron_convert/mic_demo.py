"""Live microphone transcription demo.

Captures audio from the default input device at 16 kHz mono, maintains a
rolling N-second buffer, and runs NeMo's transcribe on it once per HOP
seconds. The current utterance is rendered in place; when speech ends
(>= --silence-seconds of low-RMS audio), the line is finalized — it
scrolls off into the transcript history and we start fresh for the next
utterance.

Uses the in-process NeMo model, NOT the Core AI `.aimodel` artifact.

Run:
    python -m nemotron_convert.mic_demo
"""

from __future__ import annotations

import argparse
import logging
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
        description="Live microphone transcription via NeMo. Pause to finalize a line.",
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
        "--silence-seconds",
        type=float,
        default=1.2,
        help="Silence duration that finalizes the current utterance into the history.",
    )
    parser.add_argument(
        "--rms-threshold",
        type=float,
        default=0.003,
        help="RMS below this counts as silence (used for the silence timer).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="sounddevice input device index (default: system default).",
    )
    args = parser.parse_args()

    print(f"[mic-demo] Loading {args.model_id} …", flush=True)
    logging.getLogger("nemo_logger").setLevel(logging.WARNING)

    import nemo.collections.asr as nemo_asr
    model = nemo_asr.models.ASRModel.from_pretrained(args.model_id)
    model.eval()

    sample_rate = 16000
    window_samples = int(sample_rate * args.window_seconds)
    rolling = np.zeros(window_samples, dtype=np.float32)
    audio_queue: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()

    def audio_callback(indata, _frames, _time, status):
        if status:
            print(f"\n[mic-demo] audio status: {status}", file=sys.stderr)
        audio_queue.put(indata.copy())

    def handle_sigint(_signum, _frame):
        stop_event.set()
    signal.signal(signal.SIGINT, handle_sigint)

    print("[mic-demo] Ready. Speak into the mic. Pause = finalize. Ctrl+C to stop.")
    print()

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        device=args.device,
        blocksize=int(sample_rate * 0.1),
    )

    # Utterance state: the rendered partial is whatever was last printed on the
    # bottom line (in-place updates via \r). `since_speech` counts wall-clock
    # silence since the last time we saw audio louder than the gate. When it
    # crosses `silence-seconds` AND we have a non-empty partial, we promote
    # that partial to the scrollback as a finalized line and reset for the
    # next utterance.
    partial = ""
    finalized_count = 0
    last_speech_time = time.monotonic()
    last_finalize_time = 0.0
    had_speech_in_window = False  # has any chunk above threshold landed since last finalize?

    with stream:
        last_run = time.monotonic()
        while not stop_event.is_set():
            try:
                chunk = audio_queue.get(timeout=0.05).flatten()
            except queue.Empty:
                chunk = None

            now = time.monotonic()

            if chunk is not None:
                # Slide the rolling window: drop oldest len(chunk), append new.
                if len(chunk) >= window_samples:
                    rolling = chunk[-window_samples:]
                else:
                    rolling = np.concatenate([rolling[len(chunk):], chunk])

                # Track the latest speech timestamp using a SHORT trailing
                # window (the last hop's worth of audio), not the whole rolling
                # buffer — otherwise the "is there silence now?" answer lags
                # behind reality by ~window_seconds.
                tail_samples = int(sample_rate * args.hop_seconds)
                tail = rolling[-tail_samples:] if len(rolling) >= tail_samples else rolling
                tail_rms = float(np.sqrt(np.mean(tail ** 2)))
                if tail_rms >= args.rms_threshold:
                    last_speech_time = now
                    had_speech_in_window = True

            # Silence-driven finalization. If we have a partial and it's been
            # quiet for `silence-seconds`, commit it to scrollback.
            silence_elapsed = now - last_speech_time
            if partial and silence_elapsed >= args.silence_seconds:
                # Promote to scrollback. Erase the partial line, print the
                # finalized version with a leading newline so it stays visible
                # after the next partial overwrites the current line.
                pad = " " * max(0, len(partial) + 2)
                print(f"\r{pad}\r{partial}", flush=True)
                finalized_count += 1
                partial = ""
                had_speech_in_window = False
                # Reset the rolling buffer so the next utterance starts clean.
                # Without this, the buffer's tail of silence keeps producing
                # the same final text on subsequent transcribe calls.
                rolling = np.zeros(window_samples, dtype=np.float32)
                last_finalize_time = now
                continue

            # Time-based hop: only transcribe periodically.
            if now - last_run < args.hop_seconds:
                continue
            last_run = now

            # Skip transcribe if the rolling buffer has no recent speech.
            if not had_speech_in_window:
                continue

            try:
                result = model.transcribe(
                    [rolling.copy()],
                    batch_size=1,
                    verbose=False,
                )
            except Exception as exc:
                print(f"\n[mic-demo] transcribe failed: {exc}", file=sys.stderr)
                continue

            r0 = result[0]
            if hasattr(r0, "text"):
                text = r0.text
            elif isinstance(r0, str):
                text = r0
            else:
                text = ""
            text = (text or "").strip()

            if text and text != partial:
                pad = " " * max(0, len(partial) - len(text))
                print(f"\r{text}{pad}", end="", flush=True)
                partial = text

    # Flush a final newline. If there's a still-uncommitted partial, commit
    # it so the user sees the very last thing they said.
    if partial:
        pad = " " * max(0, len(partial) + 2)
        print(f"\r{pad}\r{partial}", flush=True)
    print(f"\n[mic-demo] stopped. {finalized_count} line(s) transcribed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
