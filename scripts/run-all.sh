#!/usr/bin/env bash
# run-all.sh — verify the Swift + Core AI build and run the bundled sample.
#
# Builds the Swift package, runs tests, and transcribes the bundled
# fixtures/sample.wav as a smoke test. No Python required.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

step() { printf '\n\033[1;36m▸ %s\033[0m\n' "$*"; }
ok()   { printf '  \033[1;32m✓\033[0m %s\n' "$*"; }
fail() { printf '  \033[1;31m✗\033[0m %s\n' "$*" >&2; exit 1; }

AIMODEL="$ROOT/fixtures/aimodel/nemotron-streaming.aimodel"
SAMPLE="$ROOT/fixtures/sample.wav"

step "1/4  Prerequisites"
command -v swiftc >/dev/null || fail "swiftc not on PATH — install Xcode 27 beta"
[ -e "$AIMODEL" ] || fail "missing $AIMODEL"
[ -e "$SAMPLE" ]  || fail "missing $SAMPLE"
ok "macOS SDK $(xcrun --sdk macosx --show-sdk-version), Swift $(swift --version 2>&1 | head -1 | awk '{print $4}')"

step "2/4  Build (release)"
swift build --configuration release >/dev/null
ok "nemotron-asr CLI built"

step "3/4  Tests"
swift test 2>&1 | tail -3
ok "tests done"

step "4/4  Sample transcription"
"$ROOT/.build/release/nemotron-asr" file "$SAMPLE" --model "$AIMODEL"

printf '\n\033[1;32m✓ run-all complete.\033[0m  Next:\n'
printf '   • Live mic:   \033[1m./scripts/demo.sh\033[0m\n'
printf '   • Any WAV:    \033[1m./scripts/demo.sh file path/to/audio.wav\033[0m\n'
