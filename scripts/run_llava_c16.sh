#!/usr/bin/env bash
# LLaVA coarse-16 follow-up: at coarse 64 the fresh VQAv2 set is near-saturated
# for LLaVA-1.6-Mistral-7B (STOP 82.7%), so re-run both prompt variants at a
# 16-token-equivalent coarse budget to re-open a perceptual bottleneck —
# same move as the Phi-4 coarse-16 run (lockbox_vqav2_phi4_c16).
set -uo pipefail
cd "$(dirname "$0")/.."

PY=$SERVE_VENV/bin/python
MODEL=llava-hf/llava-v1.6-mistral-7b-hf
PORT=8101

OTHER=""
if [[ -f logs/serve_spanconf.log ]]; then
  USED=$(grep -oP '^\[serve\] GPU \K[0-9]+' logs/serve_spanconf.log | head -1 || true)
  if [[ -n "${USED:-}" ]]; then OTHER=$(( 1 - USED )); fi
fi

echo "[llava-c16] starting server on GPU '${OTHER:-auto}' port $PORT"
PPU_MODEL=$MODEL PPU_PORT=$PORT PPU_GPU=${OTHER:-} PPU_GPU_MEM=0.35 \
  scripts/serve.sh > logs/serve_llava_c16.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for i in $(seq 1 180); do
  sleep 5
  if curl -sf localhost:$PORT/v1/models > /dev/null; then break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[llava-c16] server died"; exit 1; fi
done
curl -sf localhost:$PORT/v1/models > /dev/null || { echo "[llava-c16] server never came up"; exit 1; }
echo "[llava-c16] server up"

for variant in "" oneword; do
  tag=${variant:-orig}
  echo "[llava-c16] full run ($tag)"
  PPU_PROMPT_VARIANT=$variant $PY scripts/collect_states.py \
    --manifest data/manifests/vqav2_fresh.jsonl --ids data/vqav2_fresh_ids.txt \
    --coarse 16 --highres 192 --run-id "fresh_vqav2_llava_c16_$tag" \
    --base-url http://localhost:$PORT/v1 --model $MODEL \
    || { echo "[llava-c16] FAILED ($tag)"; exit 1; }
done

echo "[llava-c16] done; stopping server"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo LLAVA_C16_DONE
