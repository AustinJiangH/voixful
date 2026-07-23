#!/usr/bin/env bash
# demo.sh — one-command Voixful ASR demo through Swift + Core AI.
#
# Runs entirely on Apple Silicon's ANE/GPU via Core AI — no Python at runtime.
# A converted .aimodel must exist. It is NOT committed (large binary, gitignored):
# download it from Hugging Face or build it with convert/ (see convert/README.md),
# then point this script at it with $VOIXFUL_MODEL (default: the path below).
#
# Usage:
#   ./scripts/demo.sh                       live mic — speak, see transcript
#   ./scripts/demo.sh file <path-to-wav>    one-shot transcribe a WAV
#   ./scripts/demo.sh sample                transcribe the bundled fixtures/sample.wav
#   ./scripts/demo.sh --help                show CLI help

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

AIMODEL="${VOIXFUL_MODEL:-$ROOT/fixtures/aimodel/nemotron-streaming.aimodel}"
if [ ! -e "$AIMODEL" ]; then
    echo "[demo] missing model: $AIMODEL" >&2
    echo "[demo] The .aimodel is not committed. Set \$VOIXFUL_MODEL to a converted" >&2
    echo "[demo] asset, or build one with convert/ (see convert/README.md)." >&2
    exit 1
fi

# Build once; SwiftPM is fast and idempotent.
swift build --configuration release >/dev/null

CLI="$ROOT/.build/release/voixful"

case "${1:-mic}" in
    file)
        shift
        if [ "$#" -eq 0 ]; then
            echo "[demo] usage: ./scripts/demo.sh file <path-to-wav>" >&2
            exit 2
        fi
        exec "$CLI" transcribe "$1" --model "$AIMODEL"
        ;;
    sample)
        SAMPLE="$ROOT/fixtures/sample.wav"
        if [ ! -e "$SAMPLE" ]; then
            echo "[demo] $SAMPLE missing." >&2
            exit 1
        fi
        exec "$CLI" transcribe "$SAMPLE" --model "$AIMODEL"
        ;;
    --help|-h|help)
        sed -n '2,13p' "${BASH_SOURCE[0]}"
        echo
        exec "$CLI" --help
        ;;
    *)
        if [ "$(uname)" = "Darwin" ]; then
            cat <<'NOTE'
[demo] On first launch macOS will ask for microphone access for "voixful".
[demo] Grant it once; subsequent runs are silent.

NOTE
        fi
        exec "$CLI" mic --model "$AIMODEL" "$@"
        ;;
esac
