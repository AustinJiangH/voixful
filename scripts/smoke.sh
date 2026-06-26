#!/usr/bin/env bash
# smoke.sh — fast wiring sanity check. No model required.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[smoke] Swift build"
swift build >/dev/null
echo "[smoke] build: PASS"

echo "[smoke] Swift tests (model-free tests run; fixture-gated ones XCTSkip)"
swift test 2>&1 | tail -2

echo "[smoke] PASS — wiring is sane."
echo "[smoke] For an end-to-end run against the .aimodel assets, use:"
echo "[smoke]   scripts/verify-backends.sh [MODELS_DIR]"
