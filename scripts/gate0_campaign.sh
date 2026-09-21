#!/usr/bin/env bash
# Gate 0 on MME-RealWorld-Lite, unattended.
#
# Runs the ladder in order and STOPS at the first failed gate: a 200-sample
# campaign on an unverified pipeline produces numbers that look publishable and
# are not. Progress is appended to logs/campaign_status.txt so the state is
# readable at any time without attaching to the process.
#
# Launch detached (survives logout):
#   setsid nohup scripts/gate0_campaign.sh > logs/campaign.log 2>&1 < /dev/null &
set -uo pipefail
cd $PROJECT_ROOT

PY=./.venv/bin/python
M=data/manifests/mme_realworld_lite.jsonl
OCR=data/ocr_cache.json
STATUS=logs/campaign_status.txt
CONC=${PPU_CONCURRENCY:-6}
WORKERS=${PPU_OCR_WORKERS:-24}
SEED=20260816

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }
die() { say "STOP: $*"; say "CAMPAIGN_FAILED"; exit 1; }

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

say "=== Gate 0 campaign start (concurrency=$CONC, ocr workers=$WORKERS) ==="
ensure_server || die "server did not come up"

# --- stage 1: OCR pre-pass over every image the ladder will touch -----------
say "stage 1/6: OCR pre-pass"
for N in 200 20 5; do
  $PY scripts/ocr_prepass.py --manifest $M --out $OCR --workers $WORKERS \
      --n $N --seed $SEED --stratify-by category 2>&1 | tail -3 | tee -a "$STATUS"
done

COMMON="--manifest $M --seed $SEED --stratify-by category --concurrency $CONC --ocr-cache $OCR"

# --- stage 2: 5-sample smoke on the final configuration ---------------------
say "stage 2/6: 5-sample smoke (JPEG transport, concurrent, OCR pre-pass)"
rm -f logs/g0_smoke5b.jsonl
$PY scripts/run_gate1.py $COMMON --n 5 --run-id g0_smoke5b \
    --cache data/cache_opt.sqlite 2>&1 | tail -3 | tee -a "$STATUS"

say "stage 3/6: instrumentation verification"
$PY scripts/verify_smoke.py logs/g0_smoke5b.jsonl --manifest $M 2>&1 | tee -a "$STATUS"
[ "${PIPESTATUS[0]}" = "0" ] || die "smoke verification failed"

# --- stage 4: determinism, two passes with separate caches -----------------
say "stage 4/6: 20-sample determinism (two passes, separate caches)"
rm -f logs/g0_det20_a.jsonl logs/g0_det20_b.jsonl data/cache_det_a.sqlite data/cache_det_b.sqlite
$PY scripts/run_gate1.py $COMMON --n 20 --run-id g0_det20_a \
    --cache data/cache_det_a.sqlite 2>&1 | tail -2 | tee -a "$STATUS"
$PY scripts/run_gate1.py $COMMON --n 20 --run-id g0_det20_b \
    --cache data/cache_det_b.sqlite 2>&1 | tail -2 | tee -a "$STATUS"
$PY scripts/compare_runs.py logs/g0_det20_a.jsonl logs/g0_det20_b.jsonl 2>&1 | tee -a "$STATUS"
[ "${PIPESTATUS[0]}" = "0" ] || die "determinism check failed (see mismatches above)"

# --- stage 5: the campaign -------------------------------------------------
say "stage 5/6: Gate 0 n=200 stratified, dummy arm on half"
ensure_server || die "server died before the main campaign"
$PY scripts/run_gate1.py $COMMON --n 200 --run-id g0_mme200 \
    --dummy-fraction 0.5 --cache data/cache_g0.sqlite 2>&1 | tail -3 | tee -a "$STATUS"

# --- stage 6: audit --------------------------------------------------------
say "stage 6/6: audit"
$PY scripts/analyze_gate1.py logs/g0_mme200.jsonl --manifest $M \
    --n-boot 10000 --out paper/gate0_mme_audit.md \
    --json-out paper/gate0_mme_audit.json 2>&1 | tail -60 | tee -a "$STATUS"

say "CAMPAIGN_DONE -> paper/gate0_mme_audit.md"
