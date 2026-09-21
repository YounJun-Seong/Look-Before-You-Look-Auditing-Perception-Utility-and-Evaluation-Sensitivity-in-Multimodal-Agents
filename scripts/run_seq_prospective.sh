#!/usr/bin/env bash
# Sequential prospective collection driver (paper/PREREG_SEQ_PROSPECTIVE.md).
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
PORT=8100
MAN=data/manifests/textvqa_seq_prospective.jsonl

echo "[sp] OCR pre-pass"
env PPU_OCR_THREADS=4 OMP_NUM_THREADS=2 \
  $PY scripts/ocr_prepass.py --manifest $MAN --out data/ocr_cache.json --workers 12 \
  2>&1 | tail -2 || { echo "[sp] OCR pre-pass FAILED"; exit 1; }

echo "[sp] starting Qwen server (image limit 3)"
PPU_MM_LIMIT=3 PPU_GPU_MEM=${PPU_GPU_MEM:-0.35} scripts/serve.sh > logs/serve_seq_prospective.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 240); do
  sleep 5
  if curl -sf localhost:$PORT/v1/models > /dev/null; then break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[sp] server died"; exit 1; fi
done
curl -sf localhost:$PORT/v1/models > /dev/null || { echo "[sp] server never came up"; exit 1; }
echo "[sp] server up"

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
bad = [s for s, d in per.items() if d[1] != 9 or d[2] != 9]
print(f"[integrity] {sys.argv[1]}: states {len(per)}/{sys.argv[2]}, "
      f"errors {errs}, incomplete {len(bad)}")
sys.exit(0 if errs == 0 and not bad and len(per) == int(sys.argv[2]) else 1)
EOF
}

echo "[sp] smoke (3 states)"
rm -f logs/sp_smoke.jsonl
$PY scripts/collect_seq_prospective.py --run-id sp_smoke --limit 3 \
  || { echo "[sp] smoke FAILED"; exit 1; }
no_errors logs/sp_smoke.jsonl 3 || { echo "[sp] smoke integrity FAILED"; exit 1; }

echo "[sp] full 300"
$PY scripts/collect_seq_prospective.py --run-id seq_prospective \
  || { echo "[sp] full FAILED"; exit 1; }
no_errors logs/seq_prospective.jsonl 300 || { echo "[sp] full integrity FAILED"; exit 1; }

echo "[sp] done"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo SEQ_PROSPECTIVE_DONE
