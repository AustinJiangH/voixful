#!/usr/bin/env bash
# mic-test.sh — one-shot diagnosis of "voixful mic produces nothing".
#
# Removes voixful's live-capture from the question: records the mic with
# ffmpeg, checks whether the recording is actually silent, then transcribes it.
# This isolates an OS/terminal microphone-permission problem (silent recording)
# from anything in voixful (file transcription is known to work).
#
# Usage:  ./scripts/mic-test.sh

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${VOIXFUL_MODEL:-$ROOT/fixtures/aimodel/nemotron-streaming.aimodel}"
BIN="$ROOT/.build/release/voixful"
WAV="/tmp/voixful-mic-test.wav"

if [ ! -x "$BIN" ]; then
    echo "[test] building release binary first…"
    swift build -c release >/dev/null
fi

echo
echo "=== 1/3  Recording 5 seconds from the mic — SPEAK NOW ==="
if ! ffmpeg -hide_banner -loglevel error -f avfoundation -i ":0" -t 5 -ar 16000 -ac 1 -y "$WAV" || [ ! -f "$WAV" ]; then
    echo
    echo ">>> ffmpeg could NOT capture audio."
    echo ">>> Your terminal app does not have microphone permission."
    echo ">>> Fix: System Settings > Privacy & Security > Microphone ->"
    echo ">>>      enable your terminal app, then QUIT and reopen it."
    exit 1
fi

echo
echo "=== 2/3  Was the recording actually silent? ==="
VOL_LINE="$(ffmpeg -hide_banner -i "$WAV" -af volumedetect -f null - 2>&1 | grep max_volume)"
echo "    $VOL_LINE"
MAX="$(echo "$VOL_LINE" | grep -oE '\-?[0-9.]+ dB' | grep -oE '\-?[0-9.]+' | head -1)"
if [ -n "$MAX" ] && awk "BEGIN{exit !($MAX < -60)}"; then
    echo
    echo ">>> The recording is SILENT (max_volume ${MAX} dB)."
    echo ">>> The mic is not delivering audio to CLI tools in THIS terminal."
    echo ">>> This is an OS permission/device problem, NOT voixful."
    echo ">>> Fix:"
    echo ">>>   1. System Settings > Privacy & Security > Microphone -> enable your terminal app."
    echo ">>>   2. FULLY quit and reopen the terminal (TCC changes need a restart)."
    echo ">>>   3. System Settings > Sound > Input -> confirm the bar moves when you speak."
    exit 2
fi
echo "    Recording has real audio. Good — the OS/mic side is fine."

echo
echo "=== 3/3  Transcribing the recording through voixful ==="
"$BIN" transcribe "$WAV" --model "$MODEL"

echo
echo ">>> If you see your words above, the engine + pipeline are fine."
echo ">>> 'voixful mic' will work too as soon as this terminal has mic permission."
