# Pre-registration: blinded human-semantic utility audit

> **STATUS: WITHDRAWN — NOT EXECUTED (2026-08-30).** The annotator decided
> not to run the human audit in this form. No verdict was ever entered into
> the server-side sheets and `scripts/human_audit_analysis.py` was never run,
> so no result exists to report or suppress. The document is retained
> unmodified below for the record; the blinded sheets
> (`paper/human_audit_sheet_*.tsv`) remain empty and the private mapping
> (`data/human_audit/items_*.jsonl`) remains unopened.

**Fixed BEFORE any verdict is entered.** Date: 2026-08-28. Post-hoc relative
to the paper's confirmatory analyses; pre-registered relative to its own
annotation data.

## Question

Does the strict-scorer-defined perception utility on original-prompt VQAv2
survive when ONLY the evaluator changes — same model, same prompt, same
generations, same frozen confidence, human semantic scoring of the identical
outputs? TextVQA is annotated under the identical protocol as the control
regime where the audit predicts little change.

## Materials

- Generations: `logs/fresh_vqav2_orig.jsonl`, `logs/fresh_textvqa_orig.jsonl`
  (250 states each; STOP/HIGHRES × both replays; original prompt p0.1).
  Nothing is regenerated or rescored by a model.
- Unit: unique (state, whitespace/case-normalized answer) pair — identical
  strings within a state receive one verdict by construction (VQAv2 489,
  TextVQA 381 items).
- Blinding: the sheets show item_id, question, gold references (with counts),
  and the answer text only — never the arm, replay, resolution, image, or
  strict score. Item order is globally shuffled (seed 20260828). The mapping
  files in `data/human_audit/` stay closed during annotation.
- Verdicts: `C` correct / `P` partially correct / `I` incorrect /
  `A` ambiguous-reference (gold set itself contradictory or the question is
  unanswerable from the references). One annotator minimum; if a second
  annotates, disagreements are adjudicated by discussion and Cohen's κ on the
  pre-adjudication verdicts is reported.

## Analysis (frozen now)

1. Human score: y_human = 1 for C, 0 for I; primary analysis maps P → 0
   (conservative toward the strict scorer); a P → 1 sensitivity run is
   reported alongside. A-items are excluded from label construction (their
   states become AMBIGUOUS if an arm depends on them).
2. Per state and replay, y_human of the STOP and HIGHRES answers rebuilds the
   four-way label with THE SAME rule as the paper (both replays must agree;
   disagreement → AMBIGUOUS, excluded).
3. Deliverables:
   a. strict→human label transition matrix per dataset;
   b. the format-flip HELP subset (multi-word STOP R1 → one-word HIGHRES R1,
      strict HELP) reported separately: share that remains HELP vs becomes
      NEUTRAL/HARM under human scoring;
   c. frozen confidence ROC (whole-reply mean logprob, exactly as logged,
      never recomputed): AUC(C_original, U_strict) vs AUC(C_original,
      U_human), per dataset, with bootstrap CIs (3k resamples, seed 20260828);
   d. STOP/HIGHRES accuracy under human scoring (mean over replays).
4. Pre-declared expectations (reported either way): VQAv2 — a substantial
   fraction of strict format-flip HELP becomes non-HELP, and the frozen
   confidence AUC against U_human rises materially from 0.428; TextVQA —
   labels largely stable and AUC approximately unchanged from 0.786.

No verdict is edited after the analysis has been run. If annotation is done in
multiple sittings, the analysis runs once, after the last sheet is complete.
