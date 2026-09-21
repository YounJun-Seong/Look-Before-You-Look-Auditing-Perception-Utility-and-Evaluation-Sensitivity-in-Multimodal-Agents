#!/usr/bin/env bash
# Serve the frozen reader for PPU.
#
# Uses the evconflict venv (vllm 0.11.0 / torch 2.8.0+cu128) on purpose: that
# stack is the one verified against this driver (CUDA 12.8 ceiling), and the
# PPU venv is deliberately light — it only needs an HTTP client.
#
# NON-NEGOTIABLE: --no-enable-prefix-caching. Gate 1 fires ~11 near-identical
# requests per image; that is exactly the regime where evconflict measured
# prefix caching changing outputs (2026-08-12).
#
# Shared machine: pick a GPU with room and ask for a modest fraction.
#   PPU_GPU=1 PPU_GPU_MEM=0.35 scripts/serve.sh
set -euo pipefail

VENV=${PPU_VENV:-$SERVE_VENV}
MODEL=${PPU_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
PORT=${PPU_PORT:-8100}
GPU=${PPU_GPU:-}
MEM=${PPU_GPU_MEM:-0.40}
MAXLEN=${PPU_MAX_LEN:-16384}
MAXSEQ=${PPU_MAX_SEQS:-8}
MMLIMIT=${PPU_MM_LIMIT:-2}  # sequential (two-step) runs need 3

if [[ -z "$GPU" ]]; then
  # Pick the GPU with the most free memory right now. Other users' jobs appear
  # without warning, so this is a hint, not a reservation.
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
fi
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
echo "[serve] GPU $GPU has ${FREE}MiB free; requesting utilization $MEM"

export CUDA_VISIBLE_DEVICES="$GPU"
exec "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization "$MEM" \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "$MAXSEQ" \
  --limit-mm-per-prompt "{\"image\": $MMLIMIT}" \
  --no-enable-prefix-caching \
  --disable-log-requests
