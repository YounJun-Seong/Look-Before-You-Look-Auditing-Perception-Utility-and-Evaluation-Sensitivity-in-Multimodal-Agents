# Pre-registration: two-step sequential perception audit (TextVQA)

**Fixed before collection.** Date: 2026-08-31. Minimal sequential extension of
the one-step audit — the smallest setting in which "should the agent acquire
more information given what it has already learned?" is a real decision.

## Design

- Reader: frozen Qwen2.5-VL-7B, greedy seed 1234, frozen replay module
  (replays 1-2), TextVQA views (coarse 192 / highres 512).
- States: the 100-state outcome-blind multi-replay subset (sample_ids of
  `logs/mr_textvqa.jsonl`; drawn 2026-08-30 by fixed shuffle, no outcomes
  consulted). One-step references (STOP, every deterministic single arm) come
  from the existing Gate 0 log and are NOT recollected.
- New trajectories (`scripts/collect_sequential.py`, brute-force; the
  sequential decision analysis is post-hoc on the enumerated tree):
  a1 ∈ {CROP-center (grid box 0), OCR} × a2 ∈ {CROP-top-right (grid box 2),
  HIGHRES} = 4 trajectories × 2 replays × 100 states = 800 inferences.
  Evidence accumulates (HIGHRES replaces the overview; OCR text carries over);
  server runs with an image limit of 3.

## Pre-declared questions (structural; strict scorer, lenient as secondary)

1. **Sequential opportunity**: does a second acquisition add value on top of
   a1 — cross-replay oracle over {stop-after-a1, a1→CROP-TR, a1→HIGHRES}
   minus the a1 one-step value, per a1.
2. **History dependence**: does the best a2 (including stopping) differ by
   what a1 revealed — split by a1 outcome (answer changed vs unchanged vs
   correct) and report the conditional a2-value distribution.
3. **Realizability signal**: do post-a1 observables (answer change from the
   STOP answer, post-a1 confidence) predict the gain of continuing —
   ROC-AUC of "a2 helps" against these signals, cross-replay labeled.
4. Two-step vs one-step frontier: token-cost vs accuracy of the enumerated
   two-step trajectories against the one-step arms on the same states.

Interpretation is structural (does sequential value exist and depend on
acquired evidence), not a controller claim; all estimates with 10k
state-paired bootstrap CIs; failures reported as failures.

No changes after collection: states, trajectories, boxes, replays, scorers,
and the four questions above are frozen as of this date.
