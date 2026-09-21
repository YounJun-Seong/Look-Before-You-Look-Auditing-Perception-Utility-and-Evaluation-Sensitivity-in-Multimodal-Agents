#!/usr/bin/env bash
# Stronger-controller prospective collection driver
# (paper/PREREG_STRONG_CONTROLLER.md). 400 fresh TextVQA states, replays
# 1,2 (frozen protocol) + 3 (independent rewrite, for the replay-robustness
# extension). OCR pre-pass, Qwen server, 5-state smoke + integrity, full run
# + integrity. Zero-transport-error gates throughout.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
PORT=${PPU_PORT:-8100}
MAN=data/manifests/textvqa_strong.jsonl
IDS=data/textvqa_strong_ids.txt
RUN=strong_textvqa
PURPOSE="stronger-controller prospective evaluation (ridge/xgb/mlp vs mixture) + replay-3"
PREREG=paper/PREREG_STRONG_CONTROLLER.md

echo "[strong] $(date -Is) OCR pre-pass"
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  $PY scripts/ocr_prepass.py --manifest $MAN --out data/ocr_cache.json --workers 12 \
  2>&1 | tail -2 || { echo "[strong] OCR pre-pass FAILED"; exit 1; }

if curl -sf localhost:$PORT/v1/models > /dev/null; then
  echo "[strong] reusing running Qwen server"
  SRV=""
else
  echo "[strong] starting Qwen server on GPU ${PPU_GPU:-auto}"
  PPU_GPU_MEM=${PPU_GPU_MEM:-0.40} scripts/serve.sh > logs/serve_strong.log 2>&1 &
  SRV=$!
  trap '[ -n "$SRV" ] && kill $SRV 2>/dev/null' EXIT
  for i in $(seq 1 240); do
    sleep 5
    if curl -sf localhost:$PORT/v1/models > /dev/null; then break; fi
    if ! kill -0 $SRV 2>/dev/null; then echo "[strong] server died (see logs/serve_strong.log)"; exit 1; fi
  done
  curl -sf localhost:$PORT/v1/models > /dev/null || { echo "[strong] server never came up"; exit 1; }
  echo "[strong] server up"
fi

no_errors () { # log expected_states
  $PY - "$1" "$2" <<'PYEOF'
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
bad = [s for s, d in per.items() if any(d[r] != 11 for r in (1, 2, 3))]
print(f"[integrity] {sys.argv[1]}: states {len(per)}/{sys.argv[2]}, "
      f"errors {errs}, STOP-without-logprobs {noconf}, incomplete {len(bad)}")
sys.exit(0 if errs == 0 and noconf == 0 and not bad
         and len(per) == int(sys.argv[2]) else 1)
PYEOF
}

echo "[strong] smoke (5 states, replays 1,2,3)"
rm -f logs/${RUN}_smoke.jsonl
$PY scripts/collect_prospective.py --manifest $MAN --ids $IDS --replays 1,2,3 \
  --run-id ${RUN}_smoke --limit 5 --purpose "$PURPOSE (smoke)" --prereg $PREREG \
  || { echo "[strong] smoke FAILED"; exit 1; }
no_errors logs/${RUN}_smoke.jsonl 5 || { echo "[strong] smoke integrity FAILED"; exit 1; }

echo "[strong] $(date -Is) full 400"
$PY scripts/collect_prospective.py --manifest $MAN --ids $IDS --replays 1,2,3 \
  --run-id $RUN --purpose "$PURPOSE" --prereg $PREREG \
  || { echo "[strong] full FAILED"; exit 1; }
no_errors logs/$RUN.jsonl 400 || { echo "[strong] full integrity FAILED"; exit 1; }

echo "[strong] $(date -Is) done"
if [ -n "$SRV" ]; then kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; fi
echo STRONG_DONE
