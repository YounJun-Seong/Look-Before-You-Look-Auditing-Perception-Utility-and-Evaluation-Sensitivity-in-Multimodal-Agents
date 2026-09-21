#!/usr/bin/env bash
# Answer-span confidence re-collection driver: serve the frozen Qwen reader,
# re-issue STOP requests with per-token logprobs for every Qwen lockbox/fresh
# run, then stop the server. Serial requests; ~3.5k total.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=$SERVE_VENV/bin/python
mkdir -p data/span_conf logs

echo "[driver] starting server"
PPU_GPU_MEM=${PPU_GPU_MEM:-0.35} scripts/serve.sh > logs/serve_spanconf.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

for i in $(seq 1 120); do
  sleep 5
  if curl -sf localhost:8100/v1/models > /dev/null; then break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[driver] server died"; exit 1; fi
done
curl -sf localhost:8100/v1/models > /dev/null || { echo "[driver] server never came up"; exit 1; }
echo "[driver] server up"

run () { # log manifest coarse out [oneword]
  local log=$1 man=$2 coarse=$3 out=$4 variant=${5:-}
  echo "[driver] $log -> $out (variant='$variant')"
  PPU_PROMPT_VARIANT=$variant $PY scripts/collect_answer_span_conf.py \
    --log "$log" --manifest "$man" --coarse "$coarse" --out "$out" \
    || echo "[driver] FAILED: $log"
}

run logs/lockbox_vqav2.jsonl              data/manifests/vqav2_lockbox.jsonl  64  data/span_conf/lockbox_vqav2.jsonl
run logs/lockbox_textvqa.jsonl            data/manifests/textvqa_lockbox.jsonl 192 data/span_conf/lockbox_textvqa.jsonl
run logs/fresh_vqav2_orig.jsonl           data/manifests/vqav2_fresh.jsonl    64  data/span_conf/fresh_vqav2_orig.jsonl
run logs/fresh_textvqa_orig.jsonl         data/manifests/textvqa_fresh.jsonl  192 data/span_conf/fresh_textvqa_orig.jsonl
run logs/lockbox_vqav2_qwen_oneword.jsonl data/manifests/vqav2_lockbox.jsonl  64  data/span_conf/lockbox_vqav2_oneword.jsonl oneword
run logs/fresh_vqav2_oneword.jsonl        data/manifests/vqav2_fresh.jsonl    64  data/span_conf/fresh_vqav2_oneword.jsonl oneword
run logs/fresh_textvqa_oneword.jsonl      data/manifests/textvqa_fresh.jsonl  192 data/span_conf/fresh_textvqa_oneword.jsonl oneword

echo "[driver] all collections done; stopping server"
kill $SRV 2>/dev/null
wait $SRV 2>/dev/null
echo SPAN_CONF_DONE
