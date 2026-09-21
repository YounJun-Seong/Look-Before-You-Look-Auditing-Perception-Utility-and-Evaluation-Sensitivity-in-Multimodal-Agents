#!/usr/bin/env bash
# Re-collect the 183 VQAv2 held-out states poisoned by the 2026-08-28 18:18
# engine death (collect_heldout_arms.py logged APIConnectionError rollouts
# instead of aborting; those rollouts were stripped, backup
# logs/heldout_vqav2_arms.jsonl.bak_apierr). Appends under the same run-id,
# then fails loudly if any appended rollout carries an error/parse failure.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
PORT=8100

echo "[redo] starting server"
PPU_GPU_MEM=${PPU_GPU_MEM:-0.35} scripts/serve.sh > logs/serve_heldout_redo.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for i in $(seq 1 240); do
  sleep 5
  if curl -sf localhost:$PORT/v1/models > /dev/null; then break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[redo] server died"; exit 1; fi
done
curl -sf localhost:$PORT/v1/models > /dev/null || { echo "[redo] server never came up"; exit 1; }
echo "[redo] server up"

echo "[redo] vqav2 redo (183 states)"
$PY scripts/collect_heldout_arms.py --manifest data/manifests/vqav2_fresh.jsonl \
  --ids data/vqav2_fresh_ids_redo.txt --coarse 64 --run-id heldout_vqav2_arms \
  || { echo "[redo] collection FAILED"; exit 1; }

$PY - <<'EOF' || { echo "[redo] INTEGRITY FAILED"; exit 1; }
import json
from collections import defaultdict
per = defaultdict(lambda: defaultdict(int)); errs = 0
for l in open('logs/heldout_vqav2_arms.jsonl'):
    r = json.loads(l)
    if r.get('type') != 'rollout': continue
    per[r['sample_id']][r['replay']] += 1
    if r.get('error') or not r.get('parse_ok'): errs += 1
want = [x.strip() for x in open('data/vqav2_fresh_ids.txt') if x.strip()]
bad = [s for s in want if per[s][1] != 10 or per[s][2] != 10]
print(f"[integrity] states {len(want)-len(bad)}/{len(want)}, errors {errs}")
raise SystemExit(0 if not bad and errs == 0 else 1)
EOF

echo "[redo] done; stopping server"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo REDO_DONE
