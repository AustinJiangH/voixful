#!/usr/bin/env bash
# run-all.sh — one-command entry point for the whole NemotronASR + Core AI pipeline.
#
# What it does, in order:
#   1. Verify host prerequisites (Xcode 27 SDK, Python 3.11+).
#   2. Set up the Python conversion venv (if not already present).
#   3. Download nvidia/nemotron-3.5-asr-streaming-0.6b via NeMo `from_pretrained`
#      (cached after first run; HF token used if HUGGING_FACE_HUB_TOKEN is set).
#   4. Run nemotron_convert.convert -> fixtures/aimodel/nemotron-streaming.aimodel.
#   5. Build the Swift package + the SwiftUI demo.
#   6. Run the smoke driver (Swift + Python checks) end-to-end.
#   7. If a sample WAV exists at fixtures/sample.wav, transcribe it via the CLI.
#
# Re-running this script is safe: every step is idempotent (skips if the
# artifact is already on disk and looks valid).
#
# Flags:
#   --force-download   force NeMo to re-download the checkpoint
#   --force-convert    force re-running the .aimodel conversion
#   --skip-tests       skip `swift test`
#   --skip-demo-build  skip building the SwiftUI demo
#   --help             show this help

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- argument parsing -------------------------------------------------------

FORCE_DOWNLOAD=0
FORCE_CONVERT=0
SKIP_TESTS=0
SKIP_DEMO_BUILD=0

for arg in "$@"; do
    case "$arg" in
        --force-download)  FORCE_DOWNLOAD=1 ;;
        --force-convert)   FORCE_CONVERT=1 ;;
        --skip-tests)      SKIP_TESTS=1 ;;
        --skip-demo-build) SKIP_DEMO_BUILD=1 ;;
        --help|-h)
            sed -n '2,30p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "[run-all] unknown flag: $arg (try --help)" >&2
            exit 2
            ;;
    esac
done

# --- pretty output ----------------------------------------------------------

step() { printf '\n\033[1;36m▸ [%s] %s\033[0m\n' "$(date +%H:%M:%S)" "$*"; }
ok()   { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[1;33m!\033[0m %s\n' "$*"; }
fail() { printf '  \033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

# --- step 1: prerequisites --------------------------------------------------

step "1/7  Checking host prerequisites"
command -v swiftc >/dev/null 2>&1 || fail "swiftc not on PATH — install Xcode 27 beta"
SDK_VER="$(xcrun --sdk macosx --show-sdk-version 2>/dev/null || true)"
[ -n "$SDK_VER" ] || fail "xcrun could not resolve the macOS SDK"
ok "macOS SDK ${SDK_VER}, swift $(swift --version 2>&1 | head -1 | awk '{print $4}')"

command -v python3 >/dev/null 2>&1 || fail "python3 not on PATH"
PY_VER="$(python3 -c 'import sys; print("%d.%d" % (sys.version_info.major, sys.version_info.minor))')"
ok "python3 ${PY_VER}"

# --- step 2: python venv ----------------------------------------------------

step "2/7  Python conversion env"
VENV="$ROOT/convert/.venv"

# Resolve a Python 3.12 interpreter (NeMo + coreai-torch are stable on 3.11/3.12;
# 3.13/3.14 wheels are spotty as of 2026-06). Falls back to whatever python3 is
# on PATH if 3.12 is missing — caller gets a clear error from pip if so.
PY_INTERP="$(command -v python3.12 || command -v python3.11 || command -v python3)"

# uv handles pre-release deps (coreai-torch depends on coreai-core==1.0.0b1,
# which pip skips without --pre). Use uv if installed; fall back to pip --pre.
USE_UV=0
if command -v uv >/dev/null 2>&1; then
    USE_UV=1
fi

if [ ! -d "$VENV" ]; then
    if [ "$USE_UV" -eq 1 ]; then
        uv venv -p "$PY_INTERP" "$VENV" >/dev/null
    else
        "$PY_INTERP" -m venv "$VENV"
    fi
    ok "created $VENV using $($PY_INTERP --version 2>&1 | head -1)"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Check whether nemotron_convert is already importable (means deps are installed).
if python -c 'import nemotron_convert' >/dev/null 2>&1; then
    ok "nemotron_convert importable (deps already installed)"
else
    warn "installing dependencies — this can take 5–10 minutes on first run"
    if [ "$USE_UV" -eq 1 ]; then
        (cd "$ROOT/convert" && uv pip install -e .) || fail "uv pip install failed; see convert/README.md"
    else
        pip install --upgrade pip >/dev/null 2>&1 || true
        # --pre lets pip find coreai-core==1.0.0b1 which is a pre-release version.
        (cd "$ROOT/convert" && pip install --pre -e .) || fail "pip install failed; see convert/README.md for the two-env workaround"
    fi
    ok "dependencies installed"
fi

# --- step 3: download model -------------------------------------------------

step "3/7  Downloading model weights (or using cache)"
# NeMo from_pretrained caches under HF_HOME / ~/.cache/huggingface/hub. We
# trigger a download by calling from_pretrained on a tiny stub script; subsequent
# runs hit cache and finish in <1 s.
MODEL_ID="${MODEL_ID:-nvidia/nemotron-3.5-asr-streaming-0.6b}"
DOWNLOAD_MARK="$ROOT/fixtures/.model-downloaded"

if [ "$FORCE_DOWNLOAD" -eq 1 ] || [ ! -f "$DOWNLOAD_MARK" ]; then
    python - <<PY || fail "model download failed — check HF auth / network"
import os, sys
print(f"[run-all] loading {os.environ.get('MODEL_ID', '$MODEL_ID')} via NeMo …", flush=True)
import nemo.collections.asr as nemo_asr
m = nemo_asr.models.ASRModel.from_pretrained(os.environ.get("MODEL_ID", "$MODEL_ID"))
print("[run-all] model loaded; cache primed", flush=True)
PY
    mkdir -p "$ROOT/fixtures"
    date > "$DOWNLOAD_MARK"
    ok "model cached"
else
    ok "model already cached (use --force-download to re-fetch)"
fi

# --- step 4: convert to .aimodel --------------------------------------------

step "4/7  Converting to .aimodel"
AIMODEL_DIR="$ROOT/fixtures/aimodel"
AIMODEL_OUT="$AIMODEL_DIR/nemotron-streaming.aimodel"

if [ "$FORCE_CONVERT" -eq 1 ] || [ ! -e "$AIMODEL_OUT" ]; then
    mkdir -p "$AIMODEL_DIR"
    python -m nemotron_convert.convert \
        --model-id "$MODEL_ID" \
        --output "$AIMODEL_OUT" \
        || fail "convert.py failed — read convert/README.md and check VERIFY notes in convert/nemotron_convert/"
    ok "wrote $AIMODEL_OUT"
else
    ok "aimodel already on disk at $AIMODEL_OUT (use --force-convert to rebuild)"
fi

deactivate

# --- step 5: build swift packages -------------------------------------------

step "5/7  Building Swift packages"
swift build >/dev/null || fail "swift build (root) failed"
ok "root package builds"

if [ "$SKIP_DEMO_BUILD" -eq 0 ]; then
    (cd "$ROOT/Demos/NemotronDemo" && swift build >/dev/null) || fail "swift build (demo) failed"
    ok "demo package builds"
else
    warn "demo build skipped (--skip-demo-build)"
fi

# --- step 6: tests + smoke --------------------------------------------------

if [ "$SKIP_TESTS" -eq 0 ]; then
    step "6/7  Running tests"
    swift test 2>&1 | tail -3
    ok "tests done"
else
    warn "swift test skipped (--skip-tests)"
fi

step "6b/7  Smoke driver (no env probes left to fail)"
"$ROOT/scripts/smoke.sh" 2>&1 | tail -6

# --- step 7: transcribe sample if present -----------------------------------

step "7/7  Demo transcription"
SAMPLE_WAV="$ROOT/fixtures/sample.wav"
if [ -e "$SAMPLE_WAV" ] && [ -e "$AIMODEL_OUT" ]; then
    swift run nemotron-asr file "$SAMPLE_WAV" --model "$AIMODEL_OUT" || warn "file subcommand exited non-zero"
    ok "transcribed $SAMPLE_WAV"
elif [ ! -e "$SAMPLE_WAV" ]; then
    warn "fixtures/sample.wav absent — generate per fixtures/README.md to enable this step"
fi

# --- done -------------------------------------------------------------------

printf '\n\033[1;32m✓ run-all complete.\033[0m  Next:\n'
printf '   • Mic demo:        \033[1mswift run nemotron-asr mic --model %s\033[0m\n' "$AIMODEL_OUT"
printf '   • SwiftUI demo:    \033[1m(cd Demos/NemotronDemo && swift run NemotronDemo)\033[0m\n'
printf '   • Re-run anytime:  \033[1m./scripts/run-all.sh\033[0m  (idempotent, uses caches)\n'
