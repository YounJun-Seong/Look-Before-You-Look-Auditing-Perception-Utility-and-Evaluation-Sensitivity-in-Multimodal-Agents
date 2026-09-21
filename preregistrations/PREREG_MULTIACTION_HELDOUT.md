# Pre-registration: frozen multi-action policy, held-out evaluation

**Written and frozen BEFORE any held-out CROP/OCR outcome exists.**
Date: 2026-08-28. Post-hoc relative to the paper's confirmatory lockboxes;
pre-registered relative to its own evaluation data.

## Question

Can a lightweight, realizable, cost-aware multi-action controller — trained
only on development states and frozen — capture a meaningful fraction of the
audited adaptive headroom on held-out states it has never seen, under matched
perception budgets?

## Policy (frozen before evaluation)

- Value heads: per-family ridge regression (alpha=1.0) predicting
  ΔScore(a|s) = mean-replay score(a) − score(STOP), families {CROP, OCR,
  HIGHRES}; CROP target is the mean over the state's grid-crop arms
  (uniform-grid execution rule — no within-family selection).
- Features: the 16 frozen v2.2 pre-action features (`ppu/gate_features.py`),
  standardized; conf_rank = empirical CDF of STOP mean token logprob fitted on
  the TRAINING states only (mean of both replays; from
  `data/stage_a_features.jsonl`).
- Training data: Gate 0 brute-force states only (MME 196 + TextVQA 200;
  `logs/g0_mme200.jsonl`, `logs/tv_200.jsonl`). No VQAv2 multi-action outcomes
  exist anywhere, so the VQAv2 evaluation is a cross-regime transfer and is
  labeled as such.
- Decision rule: π_λ(s) = argmax over {STOP:0} ∪ {a: Δ̂(a|s) − λ·c(a|s)};
  cost axes swept separately — λ_tok over [0] ∪ logspace(−6,−2,33) with
  c = per-state mean extra visual tokens, λ_lat over [0] ∪ logspace(−5,−0.5,33)
  with c = per-state mean added latency (s).
- Freeze artifact: `models/multiaction_frozen_v1.pkl` + `.json` manifest
  (weights, scaler, conf ECDF, feature list, git SHA, seed 20260828). No refit,
  no threshold/λ selection using held-out outcomes.

## Held-out evaluation set

- `data/manifests/vqav2_fresh.jsonl` (250) and `data/manifests/textvqa_fresh.jsonl`
  (250). STOP/HIGHRES arms already exist (2026-08-23 fresh collection,
  original prompt p0.1); the CROP and OCR arms for both replays are collected
  AFTER this freeze, serial, greedy, seed 1234, same frozen reader
  (Qwen2.5-VL-7B-Instruct). CROP candidates are the deterministic grid ∪
  fallback set (`ppu.regions.build_candidates([])`, K ≤ 8, budget 256) — the
  same source='grid' population the policy's CROP expectation was trained on;
  STOP is re-issued as a determinism check against the 2026-08-23 logs.
- These states were burned for STOP-vs-HIGHRES format analyses but no policy
  or threshold was ever tuned on them; no lockbox data is touched.

## Comparisons (all evaluated once, same states, same scorer)

Always-STOP; always-HIGHRES; always-CROP (uniform grid); always-OCR;
best fixed action chosen on TRAINING data (frozen: the fixed arm with the
highest mean training score); random allocation (expectation of Look-with-
HIGHRES at rate q); confidence STOP/HIGHRES threshold sweep; frozen v2.2 gate
STOP/HIGHRES threshold sweep (its own frozen dev-ECDF normalization);
the frozen multi-action π_λ; symmetrized cross-replay oracle (bound).

## Outcomes and metrics

- Accuracy = mean over both replays of the chosen arm's logged score.
  Primary scorer: strict (as logged). Secondary on VQAv2: the lenient rescore
  of the same generations (the audited format confound lives in strict VQAv2
  labels; both are reported).
- Costs: extra visual tokens and added latency, separate axes, never summed.
- Matched-budget read-off: best sweep point with mean extra tokens ≤ B for
  B ∈ {¼, ½, 1} × the dataset's always-HIGHRES token cost, same rule for all
  policies.

## Pre-declared success criteria

1. Primary (TextVQA, strict): at some pre-declared budget B, frozen
   multi-action accuracy − best-fixed-at-B accuracy > 0 with 95% paired
   bootstrap CI excluding 0 (10k resamples, state-paired).
2. Secondary: fraction of the held-out oracle−STOP headroom recovered by the
   frozen policy at λ=0 and at the ½-budget point (target ≥ 30%, reported
   either way).
3. VQAv2 is reported under both scorers; a strict/lenient disagreement in the
   policy ranking is itself a pre-declared finding (evaluation-confound
   propagating into policy comparison).

Failures are reported as failures. One evaluation pass; no re-runs.
