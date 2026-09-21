#!/usr/bin/env bash
# Wait until some GPU has >= NEED MiB free (default 38000: 0.40 x 81.5 GB for
# vLLM + margin), then run the given driver on that GPU. Polls every 60 s.
# Usage: scripts/wait_gpu_and_run.sh scripts/run_strong_controller.sh
set -uo pipefail
cd "$(dirname "$0")/.."
DRIVER=${1:?driver script}
NEED=${NEED_MIB:-38000}
while true; do
  best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | sort -t, -k2 -nr | head -1)
  gpu=$(echo "$best" | cut -d, -f1 | tr -d ' ')
  free=$(echo "$best" | cut -d, -f2 | tr -d ' ')
  if [ "$free" -ge "$NEED" ]; then
    echo "[wait] $(date -Is) GPU $gpu has ${free}MiB free >= ${NEED}; launching $DRIVER"
    # re-check after a short settle to avoid a job that is still allocating
    sleep 30
    free2=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    if [ "$free2" -ge "$NEED" ]; then
      PPU_GPU=$gpu bash "$DRIVER"
      exit $?
    fi
    echo "[wait] $(date -Is) GPU $gpu dropped to ${free2}MiB during settle; keep waiting"
  fi
  sleep 60
done
