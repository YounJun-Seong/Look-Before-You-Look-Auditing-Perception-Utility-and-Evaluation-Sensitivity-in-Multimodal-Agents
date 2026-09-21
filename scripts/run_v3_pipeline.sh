#!/usr/bin/env bash
# Selector v3 pipeline: hidden-state extraction -> train/freeze -> (wait for
# the prereg doc to exist) -> OCR pre-pass -> serve -> smoke -> collect 300
# fresh states -> integrity. Launch when a GPU has >=30GB free.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
HFPY=$SERVE_VENV/bin/python
TSPY=$ANALYSIS_PY
PORT=8100
MAN=data/manifests/textvqa_v3_prospective.jsonl
IDS=data/textvqa_v3_prospective_ids.txt

GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
      | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
echo "[v3] using GPU $GPU for extraction"

echo "[v3] extract: dev mme"
CUDA_VISIBLE_DEVICES=$GPU $HFPY scripts/extract_hidden_states.py \
  --manifest data/manifests/mme_realworld_lite.jsonl \
  --ids data/hidden/dev_mme_ids.txt --coarse 256 \
  --out data/hidden/dev_mme.npz --device cuda:0 \
  || { echo "[v3] extraction FAILED (mme)"; exit 1; }
echo "[v3] extract: dev textvqa"
CUDA_VISIBLE_DEVICES=$GPU $HFPY scripts/extract_hidden_states.py \
  --manifest data/manifests/textvqa_val.jsonl \
  --ids data/hidden/dev_textvqa_ids.txt --coarse 192 \
  --out data/hidden/dev_textvqa.npz --device cuda:0 \
  || { echo "[v3] extraction FAILED (textvqa)"; exit 1; }
echo "[v3] extract: v3 prospective 300"
CUDA_VISIBLE_DEVICES=$GPU $HFPY scripts/extract_hidden_states.py \
  --manifest $MAN --ids $IDS --coarse 192 \
  --out data/hidden/v3_prospective.npz --device cuda:0 \
  || { echo "[v3] extraction FAILED (v3 set)"; exit 1; }

echo "[v3] train + freeze selector v3"
$TSPY scripts/train_selector_v3.py 2>&1 | grep -vE "Warning|warn|sklearn.org" \
  || { echo "[v3] training FAILED"; exit 1; }
[ -f models/selector_frozen_v3.pkl ] || { echo "[v3] freeze artifact missing"; exit 1; }

echo "[v3] waiting for paper/PREREG_SELECTOR_V3.md (prereg gate)"
for i in $(seq 1 240); do
  [ -f paper/PREREG_SELECTOR_V3.md ] && break
  sleep 30
done
[ -f paper/PREREG_SELECTOR_V3.md ] || { echo "[v3] prereg gate TIMEOUT"; exit 1; }
echo "[v3] prereg present; proceeding to collection"

echo "[v3] OCR pre-pass"
env PPU_OCR_THREADS=4 OMP_NUM_THREADS=2 \
  $PY scripts/ocr_prepass.py --manifest $MAN --out data/ocr_cache.json --workers 12 \
  2>&1 | tail -2 || { echo "[v3] OCR pre-pass FAILED"; exit 1; }

echo "[v3] starting Qwen server"
PPU_GPU_MEM=${PPU_GPU_MEM:-0.35} scripts/serve.sh > logs/serve_v3.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 240); do
  sleep 5
  if curl -sf localhost:$PORT/v1/models > /dev/null; then break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[v3] server died"; exit 1; fi
done
curl -sf localhost:$PORT/v1/models > /dev/null || { echo "[v3] server never came up"; exit 1; }
echo "[v3] server up"

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

echo "[v3] smoke (5 states)"
rm -f logs/v3_smoke.jsonl
$PY scripts/collect_prospective.py --manifest $MAN --ids $IDS \
  --run-id v3_smoke --limit 5 \
  || { echo "[v3] smoke FAILED"; exit 1; }
no_errors logs/v3_smoke.jsonl 5 || { echo "[v3] smoke integrity FAILED"; exit 1; }

echo "[v3] full 300"
$PY scripts/collect_prospective.py --manifest $MAN --ids $IDS \
  --run-id v3_prospective \
  || { echo "[v3] full FAILED"; exit 1; }
no_errors logs/v3_prospective.jsonl 300 || { echo "[v3] full integrity FAILED"; exit 1; }

echo "[v3] done"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo V3_PIPELINE_DONE
