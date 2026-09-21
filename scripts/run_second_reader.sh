#!/usr/bin/env bash
# Second-reader (Phi-4-multimodal) Gate-0-style replication driver.
# Prereg: paper/PREREG_SECOND_READER.md. Per dataset: 5-state smoke +
# verify_smoke gate, then the 150-state ladder; zero-transport-error
# integrity check after every run (2026-08-28 lesson: collect scripts log
# APIConnectionError rollouts instead of aborting).
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
PORT=8101
MODEL=microsoft/Phi-4-multimodal-instruct
SEED=20260816
CACHE=data/cache_phi4_secondreader.sqlite

if curl -sf localhost:$PORT/v1/models > /dev/null; then
  echo "[2r] reusing running Phi-4 server on :$PORT"
  SRV=""
else
  echo "[2r] starting Phi-4 server"
  scripts/serve_phi4.sh > logs/serve_phi4_secondreader.log 2>&1 &
  SRV=$!
  trap '[ -n "$SRV" ] && kill $SRV 2>/dev/null' EXIT
  for i in $(seq 1 360); do
    sleep 10
    if curl -sf localhost:$PORT/v1/models > /dev/null; then break; fi
    if ! kill -0 $SRV 2>/dev/null; then echo "[2r] server died"; exit 1; fi
  done
  curl -sf localhost:$PORT/v1/models > /dev/null || { echo "[2r] server never came up"; exit 1; }
  echo "[2r] server up"
fi

no_errors () { # log
  $PY - "$1" <<'EOF'
import json, sys
errs = bad = n = 0
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("type") != "rollout": continue
    n += 1
    if r.get("error"): errs += 1
print(f"[integrity] {sys.argv[1]}: {n} rollouts, {errs} transport errors")
sys.exit(0 if n > 0 and errs == 0 else 1)
EOF
}

run_ds () { # name manifest coarse highres fulltok stratify ocr
  local name=$1 man=$2 coarse=$3 hi=$4 full=$5 strat=$6 ocr=$7
  local common="--manifest $man --seed $SEED --coarse-tokens $coarse \
    --highres-tokens $hi --full-tokens $full --dummy-fraction 0.5 \
    --concurrency 6 --ocr-cache $ocr --cache $CACHE \
    --model $MODEL --base-url http://localhost:$PORT/v1"
  [[ -n "$strat" ]] && common="$common --stratify-by $strat"

  echo "[2r] smoke ($name)"
  rm -f "logs/p4_smoke_${name}.jsonl"
  $PY scripts/run_gate1.py $common --n 5 --run-id "p4_smoke_${name}" \
    || { echo "[2r] smoke FAILED ($name)"; exit 1; }
  no_errors "logs/p4_smoke_${name}.jsonl" || { echo "[2r] smoke integrity FAILED ($name)"; exit 1; }
  $PY scripts/verify_smoke.py "logs/p4_smoke_${name}.jsonl" --manifest "$man" \
      --coarse-tokens "$coarse" --highres-tokens "$hi" \
    || { echo "[2r] verify_smoke FAILED ($name)"; exit 1; }

  echo "[2r] full 200 ($name)"
  $PY scripts/run_gate1.py $common --n 200 --run-id "p4_${name}200" \
    || { echo "[2r] full run FAILED ($name)"; exit 1; }
  no_errors "logs/p4_${name}200.jsonl" || { echo "[2r] full integrity FAILED ($name)"; exit 1; }
}

run_ds mme data/manifests/mme_realworld_lite.jsonl 256 1280 2560 category data/ocr_cache.json
run_ds textvqa data/manifests/textvqa_val.jsonl 192 512 0 "" data/ocr_cache_textvqa.json

echo "[2r] done; stopping server"
if [ -n "$SRV" ]; then kill $SRV 2>/dev/null; wait $SRV 2>/dev/null
else pkill -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null || true
fi
echo SECOND_READER_DONE
