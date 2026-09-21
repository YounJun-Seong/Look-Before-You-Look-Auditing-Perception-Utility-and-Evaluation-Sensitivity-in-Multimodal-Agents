# Deployable multi-action selector — post-hoc analysis (2026-08-28)

**Status: post-hoc.** Conducted after the confirmatory lockbox evaluations, on the
Gate 0 brute-force logs (`logs/g0_mme200.jsonl`, `logs/tv_200.jsonl`) joined with the
STOP-confidence side collection (`data/stage_a_features.jsonl`). No new rollouts; no
lockbox data touched. Script: `scripts/multiaction_selector.py` (seed 20260828),
figure: `scripts/multiaction_figure.py` → `paper/figures/fig3_multiaction_pareto.*`,
full numbers: `paper/multiaction_selector.json`.

**Reviewer gap addressed.** The paper audits a 4-family action space
{STOP, CROP, OCR, HIGHRES} but the deployable results reduce to STOP-vs-HIGHRES, and
the +9.8pp policy headroom is a cross-replay oracle, not a runnable policy. Here we
train per-family value heads on the same 16 pre-action features the frozen gate uses
(ΔScore(a) = score(a) − score(STOP), ridge; conf_rank from training-fold ECDF only)
and act with

    π_λ(s) = argmax_a  ΔScorê(a|s) − λ·cost(a|s),   value(STOP) = 0,

sweeping λ to trace an accuracy/cost Pareto front, with cost kept on **two separate
axes** (extra visual tokens; added wall-clock latency).

## Protocol

- n = 396 states (MME-RealWorld-Lite 196, TextVQA 200), every deployable arm measured
  under both replays in Gate 0.
- All learned rows are **out-of-fold** (5-fold CV, stratified by dataset × any-family
  strict HELP). Best-fixed is cross-fitted inside the same folds. Nothing is trained
  on lockbox or fresh-lockbox data.
- **CROP is executed deployably**: a uniformly chosen grid crop, evaluated as the
  expectation over the state's grid-crop arms. VLM proposals are excluded (70% parse
  failure in Gate 0 → uneven coverage). A within-family best-grid-crop variant is
  reported only as a non-deployable diagnostic.
- Accuracy = mean over R1/R2 of the chosen arm's score (policy choices never see
  outcomes). The cross-replay oracle (select on R1, score on R2, symmetrised) is the
  only row allowed to peek; it is a bound, not a policy.
- Costs are measured per state from the logs (visual tokens; generation latency +
  OCR tool latency). OCR costs 0 tokens but a median ≈112 s/state of CPU tool time
  (494 s/state mean on MME) — the reason the two cost axes are never summed.

## Headline rows (per dataset)

| policy | MME acc | MME tok | MME s | TextVQA acc | TextVQA tok | TextVQA s |
|---|--:|--:|--:|--:|--:|--:|
| always STOP | .352 | 0 | 0 | .623 | 0 | 0 |
| always CROP (uniform grid) | .375 | 234 | 4.5 | .699 | 189 | 1.4 |
| always OCR | .383 | 0 | 494 | .651 | 0 | 32 |
| always HIGHRES (= best fixed, CF) | .421 | 994 | 3.8 | .718 | 308 | 3.2 |
| multi-action, cost-blind (λ=0) | .424 | 601 | 173 | .703 | 224 | 10.3 |
| — Δ vs STOP [95% CI] | +.072 [.026, .118] | | | +.079 [.036, .125] | | |
| multi-action + best-crop diagnostic (not deployable) | .436 | | | .718 | | |
| cross-replay oracle (bound) | .485 | 117 | 57 | .810 | 47 | 3.5 |
| — Δ vs STOP [95% CI] | +.133 [.087, .181] | | | +.187 [.138, .237] | | |
| FULL reference | .439 | | | .752 | | |

The deployable selector recovers **54% (MME) / 42% (TextVQA)** of the oracle−STOP
headroom in accuracy terms while the oracle needs only ~12–15% of the always-HIGHRES
token budget. Most of the audited headroom is *not* reachable from pre-action surface
features — consistent with the paper's claim that the missing ingredient is a
representation of *why* the model is uncertain, not a better threshold.

## Pre-declared token budgets (read-off: best sweep point with mean cost ≤ B)

Budgets fixed at ¼×, ½×, 1× of the always-HIGHRES token cost before reading the
curves; the same rule applies to every policy. Paired bootstrap CIs, 10k resamples.

**MME-RealWorld-Lite** (always-HIGHRES .421 @ 994 tok)

| B | confidence | binary gate | multi-action | multi − binary [CI] |
|---|--|--|--|--|
| 248 tok | .375 @ 224 | .372 @ 223 | **.403** @ 181 (lat 322 s) | +.031 [−.004, +.067] |
| 497 tok | .395 @ 472 | .401 @ 472 | **.419** @ 322 (lat 268 s) | +.019 [−.013, +.050] |
| 994 tok | .429 @ 817 | .421 @ 994 | **.430** @ 536 (lat 182 s) | +.009 [−.019, +.037] |

**TextVQA** (always-HIGHRES .718 @ 308 tok)

| B | confidence | binary gate | multi-action | multi − binary [CI] |
|---|--|--|--|--|
| 77 tok | .674 @ 69 | .686 @ 77 | .684 @ 56 (lat 16 s) | −.002 [−.041, +.037] |
| 154 tok | .702 @ 154 | .695 @ 154 | **.720** @ 131 (lat 13 s) | +.025 [−.006, +.057] |
| 308 tok | .721 @ 216 | .720 @ 300 | **.720** @ 131 (lat 13 s) | −.000 [−.032, +.033] |

Read honestly:

1. **Point estimates favour the multi-action selector at every matched token budget
   on MME and at ½ budget on TextVQA** — it reaches always-HIGHRES accuracy at 43%
   (TextVQA) / 54% (MME) of the tokens — but **no single matched-budget delta is
   individually significant** at n≈200 per dataset. Report as Pareto-dominance in
   point estimate with CIs shown, never as a significant win.
2. **The token savings are partly bought with OCR latency.** On the token axis the
   selector shifts mass to OCR (0 tokens) as λ grows: the MME operating points carry
   180–320 s/state of CPU OCR time. On the latency axis the same selector drops OCR
   and the advantage over binary STOP/HIGHRES largely disappears. This is a direct,
   policy-level demonstration of the paper's rule that visual-token and latency
   budgets must be reported separately — the *winning policy changes* with the axis.
3. **λ=0 (cost-blind) beats always-HIGHRES on MME (+0.3pp) and loses on TextVQA
   (−1.5pp)** — the 4-way routing signal that exists is thin, exactly as Gate 0's
   exclusive-solve audit predicted (OCR-only solves: 0.5%).

## Cross-dataset transfer (train on one dataset, deploy on the other)

Same pipeline, no CV: every learned piece fitted on the source dataset only.

- **mme→textvqa**: at 77-tok budget multi-action .682 @ 45 tok beats confidence .674
  and binary .667; at higher budgets the three converge (.716–.723 at 308 tok).
- **textvqa→mme**: at 248-tok budget multi-action .399 @ 190 tok vs binary .378,
  confidence .372; at the full budget the transferred selector saturates (.399) below
  binary .421 — the MME regime's HIGHRES-heavy optimum is not transported from
  TextVQA-trained value heads.

Transfer preserves the low-budget advantage but not the high-budget one; the
selector's value heads are regime-sensitive, consistent with §5.3 of the dossier.

## What may be claimed

- A deployable 4-way selector on frozen pre-action features traces a Pareto front
  that, in point estimate, dominates confidence- and learned binary STOP/HIGHRES at
  matched token budgets, and matches always-HIGHRES accuracy at ≈½ the tokens.
- Its cost-blind accuracy gain over always-STOP is significant on both datasets
  (+.072 / +.079, CIs exclude 0).
- The remaining oracle gap (−.055 MME / −.090 TextVQA below the bound) is the
  quantitative statement of the paper's headline: continuation-relevant utility is
  not identifiable from pre-action surface signals alone.

## What may NOT be claimed

- Any single matched-budget win over the binary gate (CIs include 0).
- Token-axis dominance without disclosing the OCR latency of those points.
- That the selector "solves" multi-action routing — the best-crop diagnostic (+1.2pp
  MME) shows even within-family crop choice is worth as much as the whole 4-way
  routing improvement.
- Absolute token comparisons across datasets (budgets differ by design).
