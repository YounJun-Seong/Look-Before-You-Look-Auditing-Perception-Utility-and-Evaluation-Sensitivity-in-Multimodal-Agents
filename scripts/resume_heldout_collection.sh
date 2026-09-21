#!/usr/bin/env bash
# Resume of run_heldout_collection.sh after the 2026-08-29 06:34 cutoff
# (session died at VQAv2 state 196/250; server went down with it).
# VQAv2: appends the 55 remaining states (data/vqav2_fresh_ids_remaining.txt,
# incl. the stripped partial vqav2/train/406142014) to logs/heldout_vqav2_arms.jsonl
# under the same run-id. VQAv2 smoke already passed on 08-28 (parse 100/100,
# STOP determinism 10/10), so it is not repeated. TextVQA then runs the full
# preregistered smoke gate + 250-state collection unchanged.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
PORT=8100

echo "[resume] starting server"
PPU_GPU_MEM=${PPU_GPU_MEM:-0.35} scripts/serve.sh > logs/serve_heldout_resume.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for i in $(seq 1 240); do
  sleep 5
  if curl -sf localhost:$PORT/v1/models > /dev/null; then break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[resume] server died"; exit 1; fi
done
curl -sf localhost:$PORT/v1/models > /dev/null || { echo "[resume] server never came up"; exit 1; }
echo "[resume] server up"

smoke_gate () { # smoke_log fresh_log
  $PY - "$1" "$2" <<'EOF'
import json, sys
from collections import defaultdict
smoke, fresh = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(smoke) if '"rollout"' in l]
ok = sum(1 for r in rows if r["parse_ok"] and (r["answer_after"] or "").strip() != "" or r["action"]["type"] == "STOP")
old = defaultdict(dict)
for l in open(fresh):
    r = json.loads(l)
    if r.get("type") == "rollout" and r["action"]["type"] == "STOP":
        old[r["sample_id"]][r["replay"]] = r["answer_after"]
stops = [r for r in rows if r["action"]["type"] == "STOP"]
det = sum(1 for r in stops if old.get(r["sample_id"], {}).get(r["replay"]) == r["answer_after"])
ocr_rows = [r for r in rows if r["action"]["type"] == "OCR"]
print(f"[smoke] {smoke}: {len(rows)} rollouts, parse-ok {ok}, "
      f"STOP determinism {det}/{len(stops)}, OCR arms {len(ocr_rows)}")
sys.exit(0 if len(rows) > 0 and det == len(stops) and len(ocr_rows) == len(stops) else 1)
EOF
}

echo "[resume] vqav2 remainder (55 states)"
$PY scripts/collect_heldout_arms.py --manifest data/manifests/vqav2_fresh.jsonl \
  --ids data/vqav2_fresh_ids_remaining.txt --coarse 64 --run-id heldout_vqav2_arms \
  || { echo "[resume] vqav2 remainder FAILED"; exit 1; }

echo "[resume] smoke (textvqa)"
$PY scripts/collect_heldout_arms.py --manifest data/manifests/textvqa_fresh.jsonl \
  --ids data/textvqa_fresh_ids.txt --coarse 192 --run-id smoke_heldout_textvqa --limit 5 \
  || { echo "[resume] smoke run FAILED (textvqa)"; exit 1; }
smoke_gate logs/smoke_heldout_textvqa.jsonl logs/fresh_textvqa_orig.jsonl \
  || { echo "[resume] smoke gate FAILED (textvqa)"; exit 1; }

echo "[resume] full run (textvqa)"
$PY scripts/collect_heldout_arms.py --manifest data/manifests/textvqa_fresh.jsonl \
  --ids data/textvqa_fresh_ids.txt --coarse 192 --run-id heldout_textvqa_arms \
  || { echo "[resume] full run FAILED (textvqa)"; exit 1; }

echo "[resume] done; stopping server"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo HELDOUT_DONE
