# Pre-registration: human semantic adjudication of strict/format-tolerant scorer disagreements

**Fixed BEFORE any verdict is entered.** Date: 2026-09-21. Post-hoc relative to
the paper's confirmatory analyses; preregistered relative to its own
annotation data. Third protocol on this question: the 08-28 blinded audit
(withdrawn 08-30) and the 09-01 proposal (declined) were never run; this
one supersedes both (user decision 09-21, reversing the 09-20 decline).
Scope is deliberately narrow: a post-hoc validation of *where the two
scorers disagree*, not a general ranking of scorers.

## Question

On the 600-state prospective TextVQA set the selector-vs-mixture contrast is
+0.9pp [−1.6, +3.4] under the strict scorer and +4.6pp [+2.4, +6.8] under
the format-tolerant (lenient) rule. Where the two scorers disagree, is the
generation semantically correct (strict false negative) or not (lenient
false positive)?

## Materials (`scripts/make_adjudication_package.py`, seed 20260921)

- Source: all 13,200 generations of `logs/prospective_textvqa.jsonl`;
  unit = unique (state, normalized-answer) item (1,584 items: 580
  both-correct, 670 both-wrong, **245 with strict = 0 and format-tolerant
  > 0**, 89 other).
- Sheet = **300 items**: 200 disagreement items (strict = 0, format-tolerant
  > 0) drawn by proportional stratified sampling over the dominant action
  family of the item's rollouts (CROP 88, HIGHRES 49, OCR 42, STOP 21), plus
  50 both-correct and 50 both-wrong controls at random; globally shuffled.
  Private mapping `data/human_adjudication/items.jsonl` (sha256[:16]
  `1384f99672bc8d65`) stays closed during annotation.
- Blinding: the sheet (`paper/human_adjudication_sheet.html`, `.tsv`) shows
  item id, image, question, reference answers with vote counts, and the
  model prediction only — never the arm, replay, resolution, policy, or
  either scorer's value.
- Instruction (verbatim on the sheet): *Judge whether the model prediction
  correctly answers the question in context. Mark Correct only when its
  overall meaning is consistent with the reference answers and does not
  introduce a contradiction or an incompatible alternative. A prediction
  that merely contains the reference token somewhere is not necessarily
  correct. Use Ambiguous when the available image / question / references
  do not support a confident judgment.*
- Verdicts: **Semantically correct (C) / Incorrect (I) / Ambiguous (A)**.
- Annotators: **three**, independent (no discussion before all exports).
  Final label = majority vote; a majority of A → ambiguous (excluded from
  the correct/incorrect rates, counted); no majority → unresolved (excluded,
  counted). Files: the filled `annotator{1,2,3}.csv` (or a TSV export of the HTML sheet), placed in `data/human_adjudication/`.

## Pre-declared analysis (`scripts/human_adjudication_analysis.py`, run once after the last export)

1. **Agreement**: raw pairwise agreement, pairwise Cohen's κ, Fleiss' κ,
   Krippendorff's α (nominal), on all 300 items.
2. **Key numbers on the 200 disagreement items** (majority C/I only):
   * P(human correct | strict = 0, format-tolerant > 0) — strict
     false-negative share, Wilson 95% CI;
   * P(human incorrect | strict = 0, format-tolerant > 0) — lenient
     false-positive share, Wilson 95% CI;
   * count of ambiguous / unresolved among the 200;
   * the same shares by dominant family and by format-tolerant value
     (1/3, 2/3, 1), descriptive.
3. **Controls**: each scorer's agreement with the majority label on the 50 +
   50 control items (checks that annotators and scorers agree where the
   scorers agree); population-weighted agreement over the three strata
   (disagreement / both-correct / both-wrong, sizes 245 / 580 / 670).
4. **Secondary — partial human-corrected rescoring** of the 600-state set:
   strict scores overridden by the majority label on the annotated items;
   unannotated disagreement items keep strict (lower bound) or lenient
   (upper bound, sensitivity). Frozen selector v2 − frozen mixture at
   ¼×/½×/1× under each scorer (paired 10k bootstrap), alongside the
   reported strict and lenient values. Only policies frozen before this set
   was collected are evaluated.
5. **Examples**: up to 10 disagreement items judged correct and all judged
   incorrect, verbatim with all three verdicts.

Pre-declared reading template (either outcome is reportable): mostly
correct → the strict scorer over-penalizes formatting (measurement claim
stands); mostly incorrect → the format-tolerant rule admits token-containing
wrong answers (claim narrowed to "scorer sensitivity", lenient results
demoted); both substantial → the two scorers make different kinds of
errors (strongest version of the measurement-sensitivity claim).

No verdict is edited after the analysis has run; if annotation is done in
sittings, the analysis runs once after the last sheet is complete.
