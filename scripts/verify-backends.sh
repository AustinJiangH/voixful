#!/usr/bin/env bash
# Verify the voixful CLI end-to-end across the model architectures whose assets
# are present (Nemotron streaming RNN-T, Parakeet full-context TDT, Cohere AED,
# Granite NAR, Canary-Qwen SALM).
#
# Usage:
#   scripts/verify-backends.sh [MODELS_DIR]
#
# MODELS_DIR (default: ~/repos/voixful-models) holds the palette4 assets, each
# self-contained (pieces.txt bundled inside the .aimodel dir). Any backend whose
# asset is missing is SKIPPED (not failed), so a partial model set still passes:
#   nemotron-speech-streaming-en-0.6b-palette4.aimodel
#   parakeet-tdt-0.6b-v3-palette4.aimodel
#   cohere-transcribe-03-2026-palette4.aimodel
#   granite-speech-4.1-2b-nar-palette4.aimodel
#   canary-qwen-2.5b-palette4.aimodel
#   whisper-base-palette4.aimodel
#   qwen3-asr-0.6b-palette4.aimodel
#   qwen3-asr-1.7b-palette4.aimodel
#
# Builds release if needed, runs each present backend on fixtures/sample.wav, and
# checks the transcript. Exits non-zero on any failure.

set -uo pipefail
cd "$(dirname "$0")/.."

Q="${1:-$HOME/repos/voixful-models}"
NEMOTRON="nemotron-speech-streaming-en-0.6b-palette4.aimodel"
PARAKEET="parakeet-tdt-0.6b-v3-palette4.aimodel"
COHERE="cohere-transcribe-03-2026-palette4.aimodel"
GRANITE="granite-speech-4.1-2b-nar-palette4.aimodel"
CANARY="canary-qwen-2.5b-palette4.aimodel"
WHISPER="whisper-base-palette4.aimodel"
QWEN3_06="qwen3-asr-0.6b-palette4.aimodel"
QWEN3_17="qwen3-asr-1.7b-palette4.aimodel"
WAV="fixtures/sample.wav"
BIN=".build/release/voixful"

[ -x "$BIN" ] || { echo "building release…"; swift build -c release >/dev/null || exit 1; }

pass=0; fail=0; skip=0
check() {  # name, output, expected-substring (case-insensitive)
  if printf '%s' "$2" | grep -qi "$3"; then
    echo "  ✅ $1"; pass=$((pass+1))
  else
    echo "  ❌ $1 → '$(printf '%s' "$2" | head -1)'"; fail=$((fail+1))
  fi
}
run() {  # name, asset, expected-substring, [extra CLI args...]
  local name="$1" asset="$2" expect="$3"; shift 3
  if [ ! -e "$Q/$asset" ]; then
    echo "  ⏭️  $name (asset $asset not found)"; skip=$((skip+1)); return
  fi
  check "$name" "$($BIN transcribe "$WAV" --model "$Q/$asset" "$@" 2>&1)" "$expect"
}

echo "voixful backend verification (models: $Q)"

check "languages" "$($BIN languages 2>&1 | tr '\n' ' ')" "en-US"
run "nemotron streaming" "$NEMOTRON" "fox"
run "nemotron json"      "$NEMOTRON" "text" --format json
run "parakeet"           "$PARAKEET" "quick brown fox"
run "cohere"             "$COHERE"   "quick brown fox"
run "granite"            "$GRANITE"  "quick brown fox"
run "canary-qwen"        "$CANARY"   "quick brown fox"
run "whisper"            "$WHISPER"  "quick brown fox"
run "qwen3-asr 0.6b"     "$QWEN3_06" "quick brown fox"
run "qwen3-asr 1.7b"     "$QWEN3_17" "quick brown fox"

echo ""
echo "TOTAL: $pass passed, $fail failed, $skip skipped"
echo "(live mic TUI: 'voixful mic --model <$NEMOTRON>' — test manually)"
[ "$fail" -eq 0 ]
