#!/usr/bin/env bash
# demo.sh — one-command Nemotron 3.5 ASR demo through Swift + Core AI.
#
# Runs entirely on Apple Silicon's ANE/GPU via Core AI — no Python at runtime.
# The .aimodel artifact at fixtures/aimodel/nemotron-streaming.aimodel must
# exist; it ships in the repo or is rebuilt by re-running the (now-archived)
# convert pipeline.
#
# Usage:
#   ./scripts/demo.sh                       live mic — speak, see transcript
#   ./scripts/demo.sh file <path-to-wav>    one-shot transcribe a WAV
#   ./scripts/demo.sh sample                transcribe the bundled fixtures/sample.wav
#   ./scripts/demo.sh --help                show CLI help

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

AIMODEL="$ROOT/fixtures/aimodel/nemotron-streaming.aimodel"
if [ ! -e "$AIMODEL" ]; then
    echo "[demo] missing $AIMODEL" >&2
    echo "[demo] the converted .aimodel ships in the repo; if it's been deleted," >&2
    echo "[demo] you'll need to regenerate it via the (archived) convert pipeline." >&2
    exit 1
fi

# Build once; SwiftPM is fast and idempotent.
swift build --configuration release >/dev/null

CLI="$ROOT/.build/release/nemotron-asr"

case "${1:-mic}" in
    file)
        shift
        if [ "$#" -eq 0 ]; then
            echo "[demo] usage: ./scripts/demo.sh file <path-to-wav>" >&2
            exit 2
        fi
        exec "$CLI" file "$1" --model "$AIMODEL"
        ;;
    sample)
        SAMPLE="$ROOT/fixtures/sample.wav"
        if [ ! -e "$SAMPLE" ]; then
            echo "[demo] $SAMPLE missing." >&2
            exit 1
        fi
        exec "$CLI" file "$SAMPLE" --model "$AIMODEL"
        ;;
    --help|-h|help)
        sed -n '2,12p' "${BASH_SOURCE[0]}"
        echo
        exec "$CLI" --help
        ;;
    *)
        if [ "$(uname)" = "Darwin" ]; then
            cat <<'NOTE'
[demo] On first launch macOS will ask for microphone access for "nemotron-asr".
[demo] Grant it once; subsequent runs are silent.

NOTE
        fi
        exec "$CLI" mic --model "$AIMODEL" "$@"
        ;;
esac
