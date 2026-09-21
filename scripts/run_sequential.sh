#!/usr/bin/env bash
# Two-step sequential collection driver (paper/PREREG_SEQUENTIAL.md).
# Needs its own server instance with a 3-image limit.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
PORT=8100

echo "[seq] starting Qwen server (image limit 3)"
PPU_MM_LIMIT=3 PPU_GPU_MEM=${PPU_GPU_MEM:-0.35} scripts/serve.sh > logs/serve_sequential.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 240); do
  sleep 5
  if curl -sf localhost:$PORT/v1/models > /dev/null; then break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[seq] server died"; exit 1; fi
done
curl -sf localhost:$PORT/v1/models > /dev/null || { echo "[seq] server never came up"; exit 1; }
echo "[seq] server up"

no_errors () { # log expected_states
  $PY - "$1" "$2" <<'EOF'
import json, sys
from collections import defaultdict
per = defaultdict(lambda: defaultdict(int)); errs = 0
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("type") != "rollout": continue
    per[r["sample_id"]][r["replay"]] += 1
    if r.get("error"): errs += 1
bad = [s for s, d in per.items() if d[1] != 4 or d[2] != 4]
print(f"[integrity] {sys.argv[1]}: states {len(per)}/{sys.argv[2]}, "
      f"errors {errs}, incomplete {len(bad)}")
sys.exit(0 if errs == 0 and not bad and len(per) == int(sys.argv[2]) else 1)
EOF
}

echo "[seq] smoke (3 states)"
rm -f logs/seq_smoke.jsonl
$PY scripts/collect_sequential.py --run-id seq_smoke --limit 3 \
  || { echo "[seq] smoke FAILED"; exit 1; }
no_errors logs/seq_smoke.jsonl 3 || { echo "[seq] smoke integrity FAILED"; exit 1; }

echo "[seq] full 100"
$PY scripts/collect_sequential.py --run-id seq_textvqa \
  || { echo "[seq] full FAILED"; exit 1; }
no_errors logs/seq_textvqa.jsonl 100 || { echo "[seq] full integrity FAILED"; exit 1; }

echo "[seq] done"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo SEQUENTIAL_DONE
