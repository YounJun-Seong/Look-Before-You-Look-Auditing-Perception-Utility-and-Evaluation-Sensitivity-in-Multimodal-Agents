# Pre-registration: second-reader multi-action replication

**Fixed before collection.** Date: 2026-08-30. Addresses the generalization
limitation that the primary multi-action audit (Gate 0) is Qwen2.5-VL-7B only.

## Design

- Reader: **microsoft/Phi-4-multimodal-instruct** (vLLM, vision LoRA, greedy,
  the same decode settings as its lockbox runs), port 8101.
- Protocol: the Gate 0 brute-force ladder (`scripts/run_gate1.py`) unchanged —
  same action set, same replay-2 paraphrase procedure, same strict scorers,
  same pixel views as the Qwen runs:
  - MME-RealWorld-Lite: coarse 256 / highres 1280 (Qwen-token-equivalent
    accounting), stratified by category, ocr `data/ocr_cache.json`.
  - TextVQA: coarse 192 / highres 512 / FULL = native,
    ocr `data/ocr_cache_textvqa.json`.
- Sample: **n = 200 states per dataset** — the full Gate 0 draw (seed
  20260816, same stratification), i.e. the identical state IDs as the Qwen
  primary audit; sampled by state ID only, never by Qwen outcomes or labels.
  Replays {1, 2}, dummy_fraction 0.5, concurrency 6.
  *Amendment 2026-08-30, before any full-run data existed (only the 5-state
  MME smoke had run): n raised from the initially drafted 150 to the full 200
  per the user's decision — identical states beat a 25% cost saving.*
- Gate: 5-state smoke + `scripts/verify_smoke.py` must pass per dataset before
  the 150-state run; a post-run integrity check requires zero transport-error
  rollouts (lesson from the 2026-08-28 engine-death incident).

## Pre-declared questions (structural, not numerical, replication)

- **H1 (primary) — state dependence**: symmetric cross-replay oracle > best
  fixed policy (cross-fitted). Primary evidence is a **positive pooled
  symmetric headroom**; dataset-specific estimates and bootstrap CIs are
  reported separately, and one dataset's CI crossing 0 is not by itself a
  failure.
- **H2 (co-primary) — non-monotonicity**: additional perception actions that
  break replay-consistent correct STOPs exist under Phi-4 (per-action and
  any-action damage, same decomposition as `paper/per_action_harm.md`).
- **H3 (co-primary) — stopping value**: the STOP-optimal share among decisive
  states is > 0 (Qwen: 34.2% / 32.1%).
- **H4 (secondary) — budgeted adaptivity**: on the same logs, the
  budget-constrained oracle frontier retains low-budget adaptive gain
  (same Lagrangian sweep as `paper/budget_oracle_frontier.md`).

## Pre-specified outputs (all produced in one analysis pass)

1. Replication table, Qwen vs Phi-4 per dataset: STOP acc, best fixed,
   cross-replay oracle, symmetric headroom [CI], any-action DMG, HIGHRES
   harm, STOP-optimal share among decisive states.
2. Oracle action distribution (STOP / CROP / OCR / HIGHRES shares) per
   reader — readers may disagree on *which* action helps while both requiring
   state-dependent allocation.
3. Per-action harm decomposition and budget frontier on the Phi-4 logs.

Analysis: `ppu/audit.py` unchanged (frozen scorers) + the per-action-harm and
budget-frontier scripts rerun on the new logs. Coarse-view saturation for
Phi-4 (as seen on VQAv2 at 64 tok) is a possible outcome on these datasets and
is reported as such, not tuned away; any coarse-budget change would be a new
collection, not a rerun.

Expected magnitudes may differ from Qwen; the pre-declared conclusion template
is: "the magnitude varies across readers, but state-dependent perception
utility and non-monotonicity persist / do not persist."
