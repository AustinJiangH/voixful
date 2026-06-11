#!/usr/bin/env bash
# demo.sh — one-command live mic transcription demo.
#
# Speak into the mic, see the transcript update in real time. Loads the
# Nemotron checkpoint via NeMo directly (path-of-least-resistance for the live
# demo); the Swift + Core AI demo path (`Demos/NemotronDemo/`,
# `nemotron-asr mic`) is the long-term target but requires the Phase 2 Swift
# surgery that's still in progress.
#
# First run will download the model (~600 MB) and Python deps (~3 GB), so
# expect 5-10 minutes on a fresh checkout. Subsequent runs reuse the cache and
# launch the mic in seconds.
#
# Usage:
#   ./scripts/demo.sh                       live mic — speak; transcript updates
#   ./scripts/demo.sh file <path-to-wav>    one-shot transcribe a file
#   ./scripts/demo.sh sample                transcribe the bundled fixtures/sample.wav
#   ./scripts/demo.sh --window-seconds 6    mic mode with custom buffer
#   ./scripts/demo.sh --help                show full mic-mode flags

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/convert"

# Ensure the Python env exists. If not, bootstrap via run-all.sh which handles
# the heavy lifting (Python 3.12 venv, uv-resolved deps, model download).
if [ ! -d ".venv" ] || ! .venv/bin/python -c "import nemotron_convert" >/dev/null 2>&1; then
    echo "[demo] Python conversion env not set up yet — running scripts/run-all.sh first to install + download."
    "$ROOT/scripts/run-all.sh"
fi

# Make sure the mic / audio-file deps are present (added after the initial pyproject.toml).
.venv/bin/python - <<'PY' 2>/dev/null || .venv/bin/pip install sounddevice soundfile scipy >/dev/null
import sounddevice  # noqa
import soundfile    # noqa
import scipy        # noqa
PY

# Subcommand routing.
case "${1:-mic}" in
    file)
        shift
        if [ "$#" -eq 0 ]; then
            echo "[demo] usage: ./scripts/demo.sh file <path-to-wav>" >&2
            exit 2
        fi
        exec .venv/bin/python -m nemotron_convert.file_demo "$@"
        ;;
    sample)
        SAMPLE="$ROOT/fixtures/sample.wav"
        if [ ! -e "$SAMPLE" ]; then
            echo "[demo] $SAMPLE missing. Generating one via macOS 'say' …"
            say -v "Samantha" -o /tmp/_demo_sample.aiff \
                "The quick brown fox jumps over the lazy dog. This is a test of the speech recognition system."
            mkdir -p "$ROOT/fixtures"
            afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/_demo_sample.aiff "$SAMPLE" >/dev/null
            rm /tmp/_demo_sample.aiff
        fi
        exec .venv/bin/python -m nemotron_convert.file_demo "$SAMPLE"
        ;;
    --help|-h|help)
        sed -n '2,16p' "${BASH_SOURCE[0]}"
        echo
        echo "Mic-mode options (passed through):"
        .venv/bin/python -m nemotron_convert.mic_demo --help | tail -n +2
        exit 0
        ;;
    *)
        # Default: live mic.
        if [ "$(uname)" = "Darwin" ]; then
            cat <<'NOTE'
[demo] On first launch macOS will ask for microphone access for "python".
[demo] Grant it once; subsequent runs are silent.

NOTE
        fi
        exec .venv/bin/python -m nemotron_convert.mic_demo "$@"
        ;;
esac
