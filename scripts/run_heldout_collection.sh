#!/usr/bin/env bash
# Held-out arm collection driver (prereg: paper/PREREG_MULTIACTION_HELDOUT.md).
# Serves the frozen Qwen reader, then per dataset: wait for the OCR pre-pass
# to cover the manifest, 5-state smoke with parse + STOP-determinism gates,
# then the full 250-state run.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
PORT=8100

echo "[heldout] starting server"
PPU_GPU_MEM=${PPU_GPU_MEM:-0.35} scripts/serve.sh > logs/serve_heldout.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for i in $(seq 1 240); do
  sleep 5
  if curl -sf localhost:$PORT/v1/models > /dev/null; then break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[heldout] server died"; exit 1; fi
done
curl -sf localhost:$PORT/v1/models > /dev/null || { echo "[heldout] server never came up"; exit 1; }
echo "[heldout] server up"

wait_ocr () { # manifest
  for i in $(seq 1 720); do
    missing=$($PY - "$1" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, ".")
from ppu.datasets import load_manifest
cache = json.loads(Path("data/ocr_cache.json").read_text()) if Path("data/ocr_cache.json").exists() else {}
print(sum(1 for s in load_manifest(sys.argv[1]) if s.image_path not in cache))
EOF
)
    if [[ "$missing" == "0" ]]; then echo "[heldout] OCR cache complete for $1"; return 0; fi
    sleep 30
  done
  echo "[heldout] OCR pre-pass never completed for $1 (missing $missing) — proceeding with fallback engine"
}

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

run_ds () { # name manifest ids coarse fresh_log
  local name=$1 man=$2 ids=$3 coarse=$4 fresh=$5
  wait_ocr "$man"
  echo "[heldout] smoke ($name)"
  $PY scripts/collect_heldout_arms.py --manifest "$man" --ids "$ids" \
    --coarse "$coarse" --run-id "smoke_heldout_$name" --limit 5 \
    || { echo "[heldout] smoke run FAILED ($name)"; exit 1; }
  smoke_gate "logs/smoke_heldout_$name.jsonl" "$fresh" \
    || { echo "[heldout] smoke gate FAILED ($name)"; exit 1; }
  echo "[heldout] full run ($name)"
  $PY scripts/collect_heldout_arms.py --manifest "$man" --ids "$ids" \
    --coarse "$coarse" --run-id "heldout_${name}_arms" \
    || { echo "[heldout] full run FAILED ($name)"; exit 1; }
}

run_ds vqav2   data/manifests/vqav2_fresh.jsonl   data/vqav2_fresh_ids.txt   64  logs/fresh_vqav2_orig.jsonl
run_ds textvqa data/manifests/textvqa_fresh.jsonl data/textvqa_fresh_ids.txt 192 logs/fresh_textvqa_orig.jsonl

echo "[heldout] done; stopping server"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo HELDOUT_DONE
