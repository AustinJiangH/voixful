#!/usr/bin/env bash
# Verify the voixful CLI end-to-end across all three model architectures
# (Nemotron streaming RNN-T, Parakeet full-context TDT, Cohere AED).
#
# Usage:
#   scripts/verify-backends.sh [MODELS_DIR]
#
# MODELS_DIR (default: /tmp/qwork) must contain the palette4 assets:
#   nemotron-palette4.aimodel  + sibling pieces.txt   (Nemotron loads the vocab
#                                                       as a sibling of the model)
#   parakeet-palette4.aimodel  + parakeet-pieces.txt
#   cohere-palette4.aimodel    + cohere-pieces.txt
#
# Builds release if needed, runs each backend on fixtures/sample.wav, and checks
# the transcript. Exits non-zero on any failure.

set -uo pipefail
cd "$(dirname "$0")/.."

Q="${1:-/tmp/qwork}"
WAV="fixtures/sample.wav"
BIN=".build/release/voixful"

[ -x "$BIN" ] || { echo "building release…"; swift build -c release >/dev/null || exit 1; }

pass=0; fail=0
check() {  # name, output, expected-substring (case-insensitive)
  if printf '%s' "$2" | grep -qi "$3"; then
    echo "  ✅ $1"; pass=$((pass+1))
  else
    echo "  ❌ $1 → '$(printf '%s' "$2" | head -1)'"; fail=$((fail+1))
  fi
}

echo "voixful backend verification (models: $Q)"

check "languages"          "$($BIN languages 2>&1 | tr '\n' ' ')" "en-US"
check "nemotron streaming" "$($BIN transcribe "$WAV" --model "$Q/nemotron-palette4.aimodel" 2>&1)" "fox"
check "nemotron json"      "$($BIN transcribe "$WAV" --model "$Q/nemotron-palette4.aimodel" --format json 2>&1)" "text"
check "parakeet"           "$($BIN transcribe "$WAV" --model "$Q/parakeet-palette4.aimodel" 2>&1)" "quick brown fox"
check "cohere"             "$($BIN transcribe "$WAV" --model "$Q/cohere-palette4.aimodel"       2>&1)" "quick brown fox"

echo ""
echo "TOTAL: $pass passed, $fail failed"
echo "(live mic TUI: 'voixful mic --model <nemotron.aimodel>' — streaming-only, test manually)"
[ "$fail" -eq 0 ]
