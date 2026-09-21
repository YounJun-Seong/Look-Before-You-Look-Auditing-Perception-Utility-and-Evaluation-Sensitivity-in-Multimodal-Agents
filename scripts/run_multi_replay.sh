#!/usr/bin/env bash
# Multi-replay robustness collection driver (paper/PREREG_MULTI_REPLAY.md).
# Qwen server, then per dataset: 3-state smoke + integrity, then the
# 100-state replays-3..5 collection + zero-transport-error check.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
PORT=8100

if curl -sf localhost:$PORT/v1/models > /dev/null; then
  echo "[mr] reusing running Qwen server on :$PORT"
  SRV=""
else
  echo "[mr] starting Qwen server"
  PPU_GPU_MEM=${PPU_GPU_MEM:-0.35} scripts/serve.sh > logs/serve_multireplay.log 2>&1 &
  SRV=$!
  trap '[ -n "$SRV" ] && kill $SRV 2>/dev/null' EXIT
  for i in $(seq 1 240); do
    sleep 5
    if curl -sf localhost:$PORT/v1/models > /dev/null; then break; fi
    if ! kill -0 $SRV 2>/dev/null; then echo "[mr] server died"; exit 1; fi
  done
  curl -sf localhost:$PORT/v1/models > /dev/null || { echo "[mr] server never came up"; exit 1; }
  echo "[mr] server up"
fi

no_errors () { # log
  $PY - "$1" <<'EOF'
import json, sys
errs = n = 0
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("type") != "rollout": continue
    n += 1
    if r.get("error"): errs += 1
print(f"[integrity] {sys.argv[1]}: {n} rollouts, {errs} transport errors")
sys.exit(0 if n > 0 and errs == 0 else 1)
EOF
}

run_ds () { # name g0log manifest coarse highres full ocr
  local name=$1 g0=$2 man=$3 c=$4 h=$5 f=$6 ocr=$7
  echo "[mr] smoke ($name)"
  rm -f "logs/mr_smoke_${name}.jsonl"
  $PY scripts/collect_multi_replay.py --g0-log "$g0" --manifest "$man" \
    --coarse "$c" --highres "$h" --full "$f" --ocr-cache "$ocr" \
    --run-id "mr_smoke_${name}" --limit 3 \
    || { echo "[mr] smoke FAILED ($name)"; exit 1; }
  no_errors "logs/mr_smoke_${name}.jsonl" || { echo "[mr] smoke integrity FAILED ($name)"; exit 1; }

  echo "[mr] full 100 ($name)"
  $PY scripts/collect_multi_replay.py --g0-log "$g0" --manifest "$man" \
    --coarse "$c" --highres "$h" --full "$f" --ocr-cache "$ocr" \
    --run-id "mr_${name}" \
    || { echo "[mr] full FAILED ($name)"; exit 1; }
  no_errors "logs/mr_${name}.jsonl" || { echo "[mr] full integrity FAILED ($name)"; exit 1; }
}

run_ds mme logs/g0_mme200.jsonl data/manifests/mme_realworld_lite.jsonl 256 1280 2560 data/ocr_cache.json
run_ds textvqa logs/tv_200.jsonl data/manifests/textvqa_val.jsonl 192 512 0 data/ocr_cache_textvqa.json

echo "[mr] done"
if [ -n "$SRV" ]; then kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; fi
echo MULTI_REPLAY_DONE
