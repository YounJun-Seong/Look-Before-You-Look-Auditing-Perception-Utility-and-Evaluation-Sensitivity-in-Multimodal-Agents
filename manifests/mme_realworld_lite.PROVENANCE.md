# mme_realworld_lite.jsonl — provenance

**Built**: 2026-08-16 · **Source**: `yifanzhang114/MME-RealWorld-Lite` (`data.zip` →
`data/MME-RealWorld-Lite.json` + `data/imgs/`), the official 50-per-task subset.
**Rows**: 1,919 · **Inclusion-eligible** (native ≥ 4× coarse budget): 1,908 (99.4%).

## Why the strata count is 39, not 43

The MME-RealWorld paper reports **43 subtasks** for the *full* split. Our numbers:

| level | count | field |
|---|--:|---|
| domain | 5 | `meta.subtask` — Remote Sensing 150, Autonomous_Driving 750, Diagram and Table 200, Monitoring 469, OCR with Complex Context 350 |
| task | **39** | `meta.category`, verbatim from the Lite JSON |
| task, case-normalized | 37 | see alias note below |

Checked against the raw annotation file: the Lite JSON itself contains **39**
distinct `Category` values, and the adapter reproduces the set and the per-value
counts **exactly** (verified 2026-08-16). The 43 → 39 difference therefore comes
from the Lite subset, **not** from adapter normalization. A reviewer reproducing
"43-task coverage" must use the full split, which is a separate 93.5 GB download.

**Case-alias artifact (documented, deliberately not fixed):** the source data
carries two pairs that differ only in capitalization —
`vehicle/counting` (31) vs `Vehicle/counting` (19), and
`person/counting` (49) vs `Person/counting` (1). Stratified sampling treats the
raw string as the stratum key, so these are 4 strata rather than 2, and the
1-sample `Person/counting` stratum is guaranteed a slot. The effect on a 200-draw
is at most a couple of samples. Both the raw `subtask` and the raw `category` are
preserved in every manifest row, so any later analysis can merge them.

## Leakage

MME-RealWorld is an **eval-only** dataset (SPEC §5.1). Every id drawn for Gate 0
is appended to `data/burned_ids.jsonl` before the campaign runs and is excluded
from the final blind test. Lite is a curated balanced subset: Gate 0 uses it to
establish **whether the phenomenon exists**, never to claim prevalence.
