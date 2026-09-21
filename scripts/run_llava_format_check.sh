#!/usr/bin/env bash
# Second-backbone format intervention (post-hoc): LLaVA-1.6-Mistral-7B on the
# fresh VQAv2 set, original vs one-word prompt, STOP/HIGHRES x both replays.
# Mirrors the fresh Qwen collection; views use the same pixel budgets
# (Qwen-token-equivalent, as with Phi-4). Ladder discipline: 5-state smoke with
# a parse gate before each full run.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=$SERVE_VENV/bin/python
MODEL=llava-hf/llava-v1.6-mistral-7b-hf
PORT=8101

# use the GPU the span-conf server is NOT on (fall back to most-free)
OTHER=""
if [[ -f logs/serve_spanconf.log ]]; then
  USED=$(grep -oP '^\[serve\] GPU \K[0-9]+' logs/serve_spanconf.log | head -1 || true)
  if [[ -n "${USED:-}" ]]; then OTHER=$(( 1 - USED )); fi
fi

echo "[llava] starting server on GPU '${OTHER:-auto}' port $PORT"
PPU_MODEL=$MODEL PPU_PORT=$PORT PPU_GPU=${OTHER:-} PPU_GPU_MEM=0.35 \
  scripts/serve.sh > logs/serve_llava.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for i in $(seq 1 180); do
  sleep 5
  if curl -sf localhost:$PORT/v1/models > /dev/null; then break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[llava] server died"; exit 1; fi
done
curl -sf localhost:$PORT/v1/models > /dev/null || { echo "[llava] server never came up"; exit 1; }
echo "[llava] server up"

head -5 data/vqav2_fresh_ids.txt > data/smoke_llava_ids.txt

smoke_gate () { # log
  $PY - "$1" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if '"rollout"' in l]
ok = sum(1 for r in rows if r["parse_ok"] and (r["answer_after"] or "").strip())
print(f"[smoke] {sys.argv[1]}: {ok}/{len(rows)} parsed, sample answers:",
      [r["answer_after"] for r in rows[:6]])
sys.exit(0 if rows and ok / len(rows) >= 0.8 else 1)
EOF
}

run () { # run_id variant ids
  local rid=$1 variant=$2 ids=$3
  PPU_PROMPT_VARIANT=$variant $PY scripts/collect_states.py \
    --manifest data/manifests/vqav2_fresh.jsonl --ids "$ids" \
    --coarse 64 --highres 192 --run-id "$rid" \
    --base-url http://localhost:$PORT/v1 --model $MODEL
}

for variant in "" oneword; do
  tag=${variant:-orig}
  echo "[llava] smoke ($tag)"
  run "smoke_llava_$tag" "$variant" data/smoke_llava_ids.txt || { echo "[llava] smoke run FAILED ($tag)"; exit 1; }
  smoke_gate "logs/smoke_llava_$tag.jsonl" || { echo "[llava] smoke gate FAILED ($tag)"; exit 1; }
  echo "[llava] full run ($tag)"
  run "fresh_vqav2_llava_$tag" "$variant" data/vqav2_fresh_ids.txt || { echo "[llava] full run FAILED ($tag)"; exit 1; }
done

echo "[llava] done; stopping server"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo LLAVA_DONE
