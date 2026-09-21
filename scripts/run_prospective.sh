#!/usr/bin/env bash
# Prospective selector-vs-mixture collection driver
# (paper/PREREG_PROSPECTIVE_SELECTOR.md). OCR pre-pass over the 600-state
# manifest, Qwen server, 5-state smoke + integrity, full 600-state run +
# integrity. Zero-transport-error gates throughout.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
PORT=8100
MAN=data/manifests/textvqa_prospective.jsonl
IDS=data/textvqa_prospective_ids.txt

echo "[pros] OCR pre-pass"
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  $PY scripts/ocr_prepass.py --manifest $MAN --out data/ocr_cache.json --workers 12 \
  2>&1 | tail -2 || { echo "[pros] OCR pre-pass FAILED"; exit 1; }

if curl -sf localhost:$PORT/v1/models > /dev/null; then
  echo "[pros] reusing running Qwen server"
  SRV=""
else
  echo "[pros] starting Qwen server"
  PPU_GPU_MEM=${PPU_GPU_MEM:-0.35} scripts/serve.sh > logs/serve_prospective.log 2>&1 &
  SRV=$!
  trap '[ -n "$SRV" ] && kill $SRV 2>/dev/null' EXIT
  for i in $(seq 1 240); do
    sleep 5
    if curl -sf localhost:$PORT/v1/models > /dev/null; then break; fi
    if ! kill -0 $SRV 2>/dev/null; then echo "[pros] server died"; exit 1; fi
  done
  curl -sf localhost:$PORT/v1/models > /dev/null || { echo "[pros] server never came up"; exit 1; }
  echo "[pros] server up"
fi

no_errors () { # log expected_states
  $PY - "$1" "$2" <<'EOF'
import json, sys
from collections import defaultdict
per = defaultdict(lambda: defaultdict(int)); errs = 0; noconf = 0
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("type") != "rollout": continue
    per[r["sample_id"]][r["replay"]] += 1
    if r.get("error"): errs += 1
    if r["action"]["type"] == "STOP" and \
            (r.get("confidence") or {}).get("mean_logprob") is None:
        noconf += 1
bad = [s for s, d in per.items() if d[1] != 11 or d[2] != 11]
print(f"[integrity] {sys.argv[1]}: states {len(per)}/{sys.argv[2]}, "
      f"errors {errs}, STOP-without-logprobs {noconf}, incomplete {len(bad)}")
sys.exit(0 if errs == 0 and noconf == 0 and not bad
         and len(per) == int(sys.argv[2]) else 1)
EOF
}

echo "[pros] smoke (5 states)"
rm -f logs/prospective_smoke.jsonl
$PY scripts/collect_prospective.py --manifest $MAN --ids $IDS \
  --run-id prospective_smoke --limit 5 \
  || { echo "[pros] smoke FAILED"; exit 1; }
no_errors logs/prospective_smoke.jsonl 5 || { echo "[pros] smoke integrity FAILED"; exit 1; }

echo "[pros] full 600"
$PY scripts/collect_prospective.py --manifest $MAN --ids $IDS \
  --run-id prospective_textvqa \
  || { echo "[pros] full FAILED"; exit 1; }
no_errors logs/prospective_textvqa.jsonl 600 || { echo "[pros] full integrity FAILED"; exit 1; }

echo "[pros] done"
if [ -n "$SRV" ]; then kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; fi
echo PROSPECTIVE_DONE
