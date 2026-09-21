#!/usr/bin/env bash
# Hidden × nonlinear controller pipeline (paper/PREREG_HIDDEN_CONTROLLER.md):
#   1. extract STOP-time hidden states for the pool sets lacking them
#      (held-out 250, prospective 600, strong 400)
#   2. train + freeze the 9 (features × head) cells on the 1,750-state pool
#   3. append the artifact sha to the prereg (gate: prereg must exist)
#   4. OCR pre-pass, Qwen server, smoke, collect 400 fresh states (replays 1,2)
#   5. extract hidden states for the fresh set, run the preregistered eval
# Expects PPU_GPU to be set (scripts/wait_gpu_and_run.sh does this).
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
HFPY=$SERVE_VENV/bin/python
TSPY=$ANALYSIS_PY
PORT=${PPU_PORT:-8100}
GPU=${PPU_GPU:?PPU_GPU must be set}
MAN=data/manifests/textvqa_hidden.jsonl
IDS=data/textvqa_hidden_ids.txt
RUN=hidden_textvqa
PREREG=paper/PREREG_HIDDEN_CONTROLLER.md

extract () { # ids out [manifest]
  [ -f "$2" ] && { echo "[hid] $2 exists, skip"; return 0; }
  CUDA_VISIBLE_DEVICES=$GPU $HFPY scripts/extract_hidden_states.py \
    --manifest "${3:-data/manifests/textvqa_val.jsonl}" --ids "$1" --coarse 192 \
    --out "$2" --device cuda:0 2>&1 | grep -vE "Warning|warn" | tail -3
  [ -f "$2" ] || { echo "[hid] extraction FAILED for $1"; exit 1; }
}

echo "[hid] $(date -Is) step 1: pool hidden states on GPU $GPU"
extract data/textvqa_fresh_ids.txt        data/hidden/heldout_textvqa.npz data/manifests/textvqa_fresh.jsonl  # held-out set is a train-split draw
extract data/textvqa_prospective_ids.txt  data/hidden/prospective_textvqa.npz
extract data/textvqa_strong_ids.txt       data/hidden/strong_textvqa.npz

if [ -f models/hidden_controllers_frozen.pkl ]; then
  echo "[hid] step 2: frozen artifact already exists (sha $($PY -c "import json;print(json.load(open('models/hidden_controllers_frozen.json'))['sha256_16'])")) — training skipped, artifact NOT retrained"
else
  echo "[hid] $(date -Is) step 2: train + freeze"
  $TSPY scripts/train_hidden_controllers.py 2>&1 | grep -vE "Warning|warn|sklearn.org" \
    || { echo "[hid] training FAILED"; exit 1; }
  [ -f models/hidden_controllers_frozen.pkl ] || { echo "[hid] freeze artifact missing"; exit 1; }
fi

[ -f $PREREG ] || { echo "[hid] prereg missing — refusing to collect"; exit 1; }
SHA=$($PY -c "import json;print(json.load(open('models/hidden_controllers_frozen.json'))['sha256_16'])")
PRIM=$($PY -c "import json;print(json.load(open('models/hidden_controllers_frozen.json'))['primary'])")
if ! grep -q "Freeze record" $PREREG; then
  printf '\n> **Freeze record (appended by the pipeline before collection, %s):** artifact sha256[:16] `%s`, primary cell `%s`. No fresh-set outcome existed at this time.\n' \
    "$(date -Is)" "$SHA" "$PRIM" >> $PREREG
fi
echo "[hid] frozen $SHA primary $PRIM"

echo "[hid] $(date -Is) step 4: OCR pre-pass"
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  $PY scripts/ocr_prepass.py --manifest $MAN --out data/ocr_cache.json --workers 12 \
  2>&1 | tail -2 || { echo "[hid] OCR pre-pass FAILED"; exit 1; }

echo "[hid] starting Qwen server on GPU $GPU"
PPU_GPU=$GPU PPU_GPU_MEM=${PPU_GPU_MEM:-0.40} scripts/serve.sh > logs/serve_hidden.log 2>&1 &
SRV=$!
trap '[ -n "$SRV" ] && kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 240); do
  sleep 5
  if curl -sf localhost:$PORT/v1/models > /dev/null; then break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[hid] server died (logs/serve_hidden.log)"; exit 1; fi
done
curl -sf localhost:$PORT/v1/models > /dev/null || { echo "[hid] server never came up"; exit 1; }
echo "[hid] server up"

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
    if r["action"]["type"] == "STOP" and (r.get("confidence") or {}).get("mean_logprob") is None:
        noconf += 1
bad = [s for s, d in per.items() if any(d[r] != 11 for r in (1, 2))]
print(f"[integrity] {sys.argv[1]}: states {len(per)}/{sys.argv[2]}, errors {errs}, "
      f"STOP-without-logprobs {noconf}, incomplete {len(bad)}")
sys.exit(0 if errs == 0 and noconf == 0 and not bad and len(per) == int(sys.argv[2]) else 1)
PYEOF
}

PURPOSE="hidden x nonlinear controller prospective evaluation"
echo "[hid] smoke (5 states)"
rm -f logs/${RUN}_smoke.jsonl
$PY scripts/collect_prospective.py --manifest $MAN --ids $IDS --replays 1,2 \
  --run-id ${RUN}_smoke --limit 5 --purpose "$PURPOSE (smoke)" --prereg $PREREG \
  || { echo "[hid] smoke FAILED"; exit 1; }
no_errors logs/${RUN}_smoke.jsonl 5 || { echo "[hid] smoke integrity FAILED"; exit 1; }

echo "[hid] $(date -Is) full 400"
$PY scripts/collect_prospective.py --manifest $MAN --ids $IDS --replays 1,2 \
  --run-id $RUN --purpose "$PURPOSE" --prereg $PREREG \
  || { echo "[hid] full FAILED"; exit 1; }
no_errors logs/$RUN.jsonl 400 || { echo "[hid] full integrity FAILED"; exit 1; }

kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=""
echo "[hid] $(date -Is) server down; step 5: fresh-set hidden states + eval"
extract $IDS data/hidden/hidden_textvqa.npz
$TSPY scripts/hidden_controller_eval.py --log logs/$RUN.jsonl --hidden data/hidden/hidden_textvqa.npz --tag hidden \
  2>&1 | grep -vE "Warning|warn" || { echo "[hid] eval FAILED"; exit 1; }
echo "[hid] $(date -Is) done"
echo HIDDEN_DONE
