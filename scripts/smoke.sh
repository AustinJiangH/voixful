#!/usr/bin/env bash
# smoke.sh — fast wiring sanity check. No model required.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[smoke] Swift build"
swift build >/dev/null
echo "[smoke] step build: PASS"

echo "[smoke] Swift smoke command (synthetic audio)"
swift run nemotron-asr smoke

echo "[smoke] Swift tests"
swift test 2>&1 | tail -2
