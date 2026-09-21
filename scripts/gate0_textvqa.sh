#!/usr/bin/env bash
# Gate 0, second dataset: TextVQA at coarse=192 (SPEC §3.1, dataset-adaptive).
#
# The question here is narrower than on MME. Four numbers decide whether the
# predictor stage is justified (user, 2026-08-16):
#   H_net > 0, P_policy > 0, OCR-only exclusive solve, >= 3 families over 10%.
#
# Launch detached:
#   setsid nohup scripts/gate0_textvqa.sh > logs/campaign_textvqa.log 2>&1 < /dev/null &
set -uo pipefail
cd $PROJECT_ROOT

PY=./.venv/bin/python
M=data/manifests/textvqa_val.jsonl
OCR=data/ocr_cache_textvqa.json
STATUS=logs/campaign_status.txt
COARSE=192
HIGHRES=512
FULLTOK=0   # <=0 = uncapped: FULL is the native view
CONC=${PPU_CONCURRENCY:-6}
WORKERS=${PPU_OCR_WORKERS:-24}
SEED=20260816

say() { echo "[$(date +%H:%M:%S)] TEXTVQA $*" | tee -a "$STATUS"; }
die() { say "STOP: $*"; say "TEXTVQA_FAILED"; exit 1; }

ensure_server() {
  curl -s -m 5 http://localhost:8100/v1/models >/dev/null 2>&1 && return 0
  say "server down; restarting"
  PPU_GPU=${PPU_GPU:-1} PPU_GPU_MEM=${PPU_GPU_MEM:-0.35} \
    setsid nohup scripts/serve.sh >> logs/serve.log 2>&1 < /dev/null &
  for _ in $(seq 1 90); do
    sleep 20
    curl -s -m 5 http://localhost:8100/v1/models >/dev/null 2>&1 && { say "server up"; return 0; }
  done
  return 1
}

say "=== start (coarse=$COARSE, highres=$HIGHRES, full=native, concurrency=$CONC) ==="
[ -f "$M" ] || die "manifest missing: $M"
ensure_server || die "server did not come up"

COMMON="--manifest $M --seed $SEED --concurrency $CONC --ocr-cache $OCR \
        --coarse-tokens $COARSE --highres-tokens $HIGHRES --full-tokens $FULLTOK"

say "stage 1/6: OCR pre-pass"
for N in 200 5; do
  $PY scripts/ocr_prepass.py --manifest $M --out $OCR --workers $WORKERS \
      --n $N --seed $SEED 2>&1 | tail -2 | tee -a "$STATUS"
done

say "stage 2/6: 5-sample smoke (192 -> 512 -> native)"
rm -f logs/tv_smoke5.jsonl
$PY scripts/run_gate1.py $COMMON --n 5 --run-id tv_smoke5 \
    --cache data/cache_tv.sqlite 2>&1 | tail -2 | tee -a "$STATUS"

say "stage 3/6: instrumentation verification"
$PY scripts/verify_smoke.py logs/tv_smoke5.jsonl --manifest $M \
    --coarse-tokens $COARSE --crop-tokens 256 --highres-tokens $HIGHRES \
    --min-gap 0.20 2>&1 | tee -a "$STATUS"
[ "${PIPESTATUS[0]}" = "0" ] || die "smoke verification failed"

say "stage 4/6: 20-sample determinism (two passes, separate caches)"
rm -f logs/tv_det20_a.jsonl logs/tv_det20_b.jsonl data/cache_tv_det_a.sqlite data/cache_tv_det_b.sqlite
$PY scripts/run_gate1.py $COMMON --n 20 --run-id tv_det20_a \
    --cache data/cache_tv_det_a.sqlite 2>&1 | tail -2 | tee -a "$STATUS"
$PY scripts/run_gate1.py $COMMON --n 20 --run-id tv_det20_b \
    --cache data/cache_tv_det_b.sqlite 2>&1 | tail -2 | tee -a "$STATUS"
$PY scripts/compare_runs.py logs/tv_det20_a.jsonl logs/tv_det20_b.jsonl 2>&1 | tee -a "$STATUS"
[ "${PIPESTATUS[0]}" = "0" ] || die "determinism check failed"
$PY scripts/unstable_samples.py logs/tv_det20_a.jsonl logs/tv_det20_b.jsonl \
    --out data/unstable_samples_textvqa.json 2>&1 | tail -6 | tee -a "$STATUS"

say "stage 5/6: n=200, dummy arm on half"
$PY scripts/run_gate1.py $COMMON --n 200 --run-id tv_200 \
    --dummy-fraction 0.5 --cache data/cache_tv.sqlite 2>&1 | tail -3 | tee -a "$STATUS"

say "stage 6/6: audit"
$PY scripts/analyze_gate1.py logs/tv_200.jsonl --manifest $M \
    --n-boot 10000 --strata dataset --out paper/gate0_textvqa_audit.md \
    --json-out paper/gate0_textvqa_audit.json 2>&1 | tail -50 | tee -a "$STATUS"

say "TEXTVQA_DONE -> paper/gate0_textvqa_audit.md"
