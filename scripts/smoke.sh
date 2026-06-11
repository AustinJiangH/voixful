#!/usr/bin/env bash
# Phase K — Top-level smoke driver for the NemotronASR scaffold.
#
# Runs every wiring check available in this environment, reports each step's
# state (PASS / FAIL / SKIPPED), and exits non-zero if any step failed.
#
# A smoke test is the smallest-possible end-to-end exercise of the system that
# confirms "the wiring is sane" — it does NOT need real model artifacts to be
# useful. Steps gracefully SKIP when the prereqs are absent (Xcode 27 beta still
# installing, Python venv not set up, .aimodel not generated) rather than
# failing — half-installed environments are the expected state for this repo
# until PLAN.md §7 Phase 1 completes.
#
# Designed to complete in under 10 seconds when the env is ready (Steps B-D
# are the bulk of the time; Step E adds a few hundred ms; Step F is a print).
#
# Bash 3.2-compatible (macOS default): no mapfile, no readarray, no <<<.

set -euo pipefail

# Resolve project root from the script's own location so the script works
# regardless of cwd. Bash 3.2-compatible: no realpath / readlink -f.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

echo "[smoke] Nemotron ASR + Core AI — scaffold smoke driver"
echo "[smoke] PROJECT_ROOT=${PROJECT_ROOT}"

STEPS_PASSED=0
STEPS_FAILED=0
STEPS_SKIPPED=0

# step <letter> <state> <detail>
# Bash 3.2: don't use associative arrays.
step() {
    local letter="$1"
    local state="$2"
    local detail="$3"
    echo "[smoke] step ${letter}: ${state} ${detail}"
    case "${state}" in
        PASS) STEPS_PASSED=$((STEPS_PASSED + 1)) ;;
        FAIL) STEPS_FAILED=$((STEPS_FAILED + 1)) ;;
        SKIPPED) STEPS_SKIPPED=$((STEPS_SKIPPED + 1)) ;;
    esac
}

# ---------------------------------------------------------------------------
# Step A: Confirm prereqs.
# ---------------------------------------------------------------------------

HAVE_SWIFTC=0
if command -v xcrun >/dev/null 2>&1; then
    if xcrun --find swiftc >/dev/null 2>&1; then
        HAVE_SWIFTC=1
    fi
fi

HAVE_PYTHON=0
if command -v python3 >/dev/null 2>&1; then
    HAVE_PYTHON=1
fi

if [ "${HAVE_SWIFTC}" -eq 1 ] && [ "${HAVE_PYTHON}" -eq 1 ]; then
    step "A" "PASS" "swiftc + python3 both available"
elif [ "${HAVE_SWIFTC}" -eq 1 ]; then
    step "A" "SKIPPED" "swiftc available; python3 NOT found (will skip Python steps)"
elif [ "${HAVE_PYTHON}" -eq 1 ]; then
    step "A" "SKIPPED" "python3 available; swiftc NOT found (will skip Swift steps; install Xcode 27 beta)"
else
    step "A" "SKIPPED" "neither swiftc nor python3 found"
fi

# ---------------------------------------------------------------------------
# Step B: swift build. Fails fast on compile errors.
# ---------------------------------------------------------------------------

if [ "${HAVE_SWIFTC}" -eq 1 ]; then
    cd "${PROJECT_ROOT}"
    if swift build 2>&1 | tail -100; then
        step "B" "PASS" "swift build completed"
    else
        step "B" "FAIL" "swift build returned non-zero exit"
    fi
else
    step "B" "SKIPPED" "Xcode 27 not installed (no swiftc on PATH)"
fi

# ---------------------------------------------------------------------------
# Step C: swift test. Runs unconditional unit tests (ChunkSize, Language, Smoke).
# ---------------------------------------------------------------------------

if [ "${HAVE_SWIFTC}" -eq 1 ]; then
    cd "${PROJECT_ROOT}"
    if swift test 2>&1 | tail -100; then
        step "C" "PASS" "swift test completed (fixture-dependent tests will have XCTSkip'd)"
    else
        step "C" "FAIL" "swift test returned non-zero exit"
    fi
else
    step "C" "SKIPPED" "Xcode 27 not installed (no swiftc on PATH)"
fi

# ---------------------------------------------------------------------------
# Step D: `swift run nemotron-asr smoke` — exercise the Swift smoke command
# without a model. Steps 1-3 run; Step 4 reports SKIPPED.
# ---------------------------------------------------------------------------

if [ "${HAVE_SWIFTC}" -eq 1 ]; then
    cd "${PROJECT_ROOT}"
    if swift run nemotron-asr smoke 2>&1 | tail -50; then
        step "D" "PASS" "nemotron-asr smoke (model-free) completed"
    else
        step "D" "FAIL" "nemotron-asr smoke exited non-zero"
    fi
else
    step "D" "SKIPPED" "Xcode 27 not installed"
fi

# ---------------------------------------------------------------------------
# Step E: Python conversion-pipeline smoke. SKIPPED if no venv available or
# the package can't be imported.
# ---------------------------------------------------------------------------

if [ "${HAVE_PYTHON}" -eq 1 ]; then
    cd "${PROJECT_ROOT}/convert"
    # Prefer a colocated .venv if present; otherwise fall back to plain python3
    # and let the import error surface as SKIPPED rather than FAIL.
    PYTHON_CMD="python3"
    if [ -x "${PROJECT_ROOT}/convert/.venv/bin/python" ]; then
        PYTHON_CMD="${PROJECT_ROOT}/convert/.venv/bin/python"
    fi
    # Run via -m so package-relative imports resolve. If the package can't be
    # found we treat that as SKIPPED (env not set up), not FAIL.
    if "${PYTHON_CMD}" -c "import nemotron_convert" >/dev/null 2>&1; then
        if "${PYTHON_CMD}" -m nemotron_convert.smoke 2>&1 | tail -50; then
            step "E" "PASS" "nemotron_convert.smoke completed"
        else
            step "E" "FAIL" "nemotron_convert.smoke exited non-zero"
        fi
    else
        step "E" "SKIPPED" "nemotron_convert not importable (run 'pip install -e .' in convert/ — see convert/README.md)"
    fi
else
    step "E" "SKIPPED" "python3 not available"
fi

# ---------------------------------------------------------------------------
# Step F: Summary.
# ---------------------------------------------------------------------------

echo ""
echo "[smoke] summary: ${STEPS_PASSED} passed, ${STEPS_FAILED} failed, ${STEPS_SKIPPED} skipped"

if [ "${STEPS_FAILED}" -gt 0 ]; then
    step "F" "FAIL" "${STEPS_FAILED} step(s) failed"
    exit 1
fi
step "F" "PASS" "no failures"
exit 0
