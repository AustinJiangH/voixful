#!/usr/bin/env bash
# run-all.sh — verify the Swift + Core AI build end to end.
#
# Builds the Swift package, runs the test suite, and (if model assets are
# available) runs the three-backend transcription check via verify-backends.sh.
# No Python required.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

step() { printf '\n\033[1;36m▸ %s\033[0m\n' "$*"; }
ok()   { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
fail() { printf '  \033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

# Model assets live outside the repo (large). Override with $VOIXFUL_MODELS_DIR.
MODELS_DIR="${VOIXFUL_MODELS_DIR:-$HOME/repos/voixful-models}"

step "1/4  Prerequisites"
command -v swiftc >/dev/null || fail "swiftc not on PATH — install Xcode 27 beta"
ok "macOS SDK $(xcrun --sdk macosx --show-sdk-version), Swift $(swift --version 2>&1 | head -1 | awk '{print $4}')"

step "2/4  Build (release)"
swift build --configuration release >/dev/null
ok "voixful CLI built"

step "3/4  Tests"
swift test 2>&1 | tail -3
ok "tests done"

step "4/4  Three-backend transcription"
if [ -d "$MODELS_DIR" ]; then
  "$ROOT/scripts/verify-backends.sh" "$MODELS_DIR"
else
  printf '  (skipped — no model assets at %s; set VOIXFUL_MODELS_DIR)\n' "$MODELS_DIR"
fi

printf '\n\033[1;32m✓ run-all complete.\033[0m  Next:\n'
printf '   • Live mic:   \033[1mvoixful mic --model <model.aimodel>\033[0m\n'
printf '   • Any WAV:    \033[1mvoixful transcribe path/to/audio.wav --model <model.aimodel>\033[0m\n'
