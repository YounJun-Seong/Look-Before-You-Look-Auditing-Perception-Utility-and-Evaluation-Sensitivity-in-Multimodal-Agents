#!/usr/bin/env bash
# Serve Phi-4-multimodal for the second-reader replication
# (paper/PREREG_SECOND_READER.md). Reproduces the 2026-08-22 lockbox serving
# config recorded in logs/serve_phi4.log: vision LoRA, no prefix caching.
set -euo pipefail

VENV=${PPU_VENV:-$SERVE_VENV}
MODEL=microsoft/Phi-4-multimodal-instruct
PORT=${PPU_PORT:-8101}
GPU=${PPU_GPU:-}
MEM=${PPU_GPU_MEM:-0.45}

SNAP=$("$VENV/bin/python" - <<'EOF'
from huggingface_hub import snapshot_download
print(snapshot_download("microsoft/Phi-4-multimodal-instruct", local_files_only=True))
EOF
)
[ -d "$SNAP/vision-lora" ] || { echo "[serve_phi4] vision-lora missing under $SNAP"; exit 1; }

if [[ -z "$GPU" ]]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
fi
echo "[serve_phi4] GPU $GPU, snapshot $SNAP"

export CUDA_VISIBLE_DEVICES="$GPU"
exec "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --port "$PORT" \
  --trust-remote-code \
  --gpu-memory-utilization "$MEM" \
  --max-model-len 16384 \
  --max-num-seqs 8 \
  --limit-mm-per-prompt '{"image": 2}' \
  --no-enable-prefix-caching \
  --enable-lora \
  --max-lora-rank 320 \
  --lora-modules "vision=$SNAP/vision-lora" \
  --disable-log-requests
