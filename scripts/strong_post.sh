#!/usr/bin/env bash
# Wait for the stronger-controller collection to finish (STRONG_DONE in the
# driver log), then run the preregistered evaluation and the replay-triangle
# analysis once. Exits non-zero if the driver fails.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/strong_controller_driver.log
APY=$ANALYSIS_PY
while true; do
  if grep -q "STRONG_DONE" "$LOG" 2>/dev/null; then break; fi
  if grep -q "FAILED\|server died\|never came up" "$LOG" 2>/dev/null; then
    echo "[post] $(date -Is) driver reported failure; not evaluating"; exit 1
  fi
  sleep 120
done
echo "[post] $(date -Is) collection done; evaluating"
$APY scripts/strong_controller_eval.py --log logs/strong_textvqa.jsonl --tag strong 2>&1 | grep -v -i warn
$APY scripts/replay_pairwise_robustness.py --log logs/strong_textvqa.jsonl --tag strong
echo "[post] $(date -Is) STRONG_POST_DONE"
