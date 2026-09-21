#!/usr/bin/env python3
"""Optimal state-independent randomized fixed-mixture baseline (post-hoc).

The held-out cost-matched comparator is deterministic (frozen fixed arm if
affordable, else always-STOP). The strongest state-independent comparator is
instead the randomized mixture

    max_p  sum_a p(a) ybar(a)   s.t.  sum_a p(a) cbar(a) <= B,  sum_a p(a) = 1

over the fixed arms {STOP, each grid CROP slot, CROP-uniform, OCR, HIGHRES}.
With one budget constraint the optimum mixes at most two arms — the upper
concave envelope of the (mean token cost, mean accuracy) points, evaluated at
B. Mixture weights are fitted on the full evaluation sample itself, which
favors the BASELINE (no cross-fitting), so the reported adaptive advantage is
conservative. Costs follow the paper's convention: extra visual tokens
(OCR = 0 tok; its CPU latency is the separate axis and is reported).

Per-state expected mixture score sum_a p(a) y_i(a) gives a paired comparison
against the frozen multi-action selector's per-state scores at the same
matched-budget read-off (10k state-paired bootstrap, module seed).

No new inference: reuses the frozen heldout eval machinery and logs.
Outputs paper/randomized_mixture_baseline.{json,md}.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# sklearn 1.9.0-pickle shim (see scratchpad runner used for the frozen eval)
import sklearn.linear_model as lm  # noqa: E402

_orig = lm.LogisticRegression.predict_proba


def _patched(self, X):
    if not hasattr(self, "multi_class"):
        self.multi_class = "deprecated"
    return _orig(self, X)


lm.LogisticRegression.predict_proba = _patched

spec = importlib.util.spec_from_file_location(
    "hme", ROOT / "scripts/heldout_multiaction_eval.py")
hme = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hme)  # __name__ != "__main__": main() does not run

FROZEN = json.loads((ROOT / "paper/heldout_multiaction.json").read_text())
BUDGET_FRACS = (0.25, 0.5, 1.0)


def arm_menu(states, scorer):
    """State-independent fixed arms -> per-state score vec + per-state cost."""
    arms = {}
    n_slots = min(len(s[scorer]["grid"]) for s in states)
    y = lambda f: np.array([float(np.mean(s[scorer][f])) for s in states])
    arms["STOP"] = (y("stop"), np.zeros(len(states)))
    arms["HIGHRES"] = (y("hi"),
                       np.array([s["cost_vtok"]["HIGHRES"] for s in states]))
    arms["OCR"] = (y("ocr"), np.array([s["cost_vtok"]["OCR"] for s in states]))
    arms["CROP_uniform"] = (y("crop"),
                            np.array([s["cost_vtok"]["CROP"] for s in states]))
    for k in range(n_slots):
        arms[f"CROP_slot{k}"] = (
            np.array([float(np.mean(s[scorer]["grid"][k])) for s in states]),
            np.array([float(np.mean([r["cost"]["visual_tokens"]
                                     for r in s["recs"]["crops"][k]]))
                      for s in states]))
    return arms


def best_mixture(arms, B):
    """LP optimum = best single feasible arm or best 2-arm mixture on the
    budget line; exhaustive over pairs (menu is tiny)."""
    names = sorted(arms)
    ybar = {a: float(arms[a][0].mean()) for a in names}
    cbar = {a: float(arms[a][1].mean()) for a in names}
    best = None
    for a in names:  # pure arms
        if cbar[a] <= B + 1e-9:
            cand = ({a: 1.0}, ybar[a])
            if best is None or cand[1] > best[1]:
                best = cand
    for i, a in enumerate(names):  # 2-arm mixtures pinned to the budget
        for b in names[i + 1:]:
            lo, hi = sorted((a, b), key=lambda x: cbar[x])
            if not (cbar[lo] <= B + 1e-9 < cbar[hi]):
                continue
            w = (B - cbar[lo]) / (cbar[hi] - cbar[lo])  # weight on hi
            val = (1 - w) * ybar[lo] + w * ybar[hi]
            if val > best[1]:
                best = ({lo: 1 - w, hi: w}, val)
    return best


def main():
    out = {}
    md = ["# Optimal state-independent randomized fixed mixture vs frozen selector",
          "",
          "max_p sum p(a)ybar(a) s.t. sum p(a)cbar(a) <= B over fixed arms",
          "{STOP, grid CROP slots, uniform CROP, OCR, HIGHRES}; weights fitted",
          "on the evaluation sample itself (favors the baseline). Paired",
          "bootstrap of (frozen multi-action - mixture), 10k resamples.", ""]
    for ds, fresh_log, arms_log in hme.DS:
        states, _ = hme.load_states(ds, fresh_log, arms_log)
        X = hme.feats_frozen(states)
        dhat = {f: hme.POL["heads"][f].predict(X) for f in hme.FAMILIES}
        scorers = ["strict"] + (["lenient"] if ds == "vqav2" else [])
        for scorer in scorers:
            sweeps = hme.sweep_multi(states, dhat, scorer)["multiaction_vtok"]
            hi_tok = float(np.mean([s["cost_vtok"]["HIGHRES"] for s in states]))
            arms = arm_menu(states, scorer)
            frozen_bt = {round(p["budget_frac"], 2): p for p in
                         FROZEN["datasets"][ds]["scorers"][scorer]["budget_table"]}
            rows = []
            md += [f"## {ds} / {scorer}  (n={len(states)}, "
                   f"always-HIGHRES {hi_tok:.0f} tok)", "",
                   "| budget | mixture (weights, in-sample) | mix acc | mix acc (CF) | multi acc | "
                   "multi − mix [CI] | multi − mix-CF [CI] |", "|---|---|--:|--:|--:|:--|:--|"]
            for frac in BUDGET_FRACS:
                B = frac * hi_tok
                weights, mix_acc = best_mixture(arms, B)
                mix_scores = sum(w * arms[a][0] for a, w in weights.items())
                # cross-fitted mixture: weights from one fold, applied to the
                # other (2-fold by state index parity, as best_fixed_cf)
                idx = np.arange(len(states))
                cf_scores = np.zeros(len(states))
                cf_w = []
                for fold in (0, 1):
                    tr, te = idx[idx % 2 == fold], idx[idx % 2 != fold]
                    arms_tr = {a: (arms[a][0][tr], arms[a][1][tr]) for a in arms}
                    w_tr, _ = best_mixture(arms_tr, B)
                    cf_w.append(w_tr)
                    cf_scores[te] = sum(w * arms[a][0][te]
                                        for a, w in w_tr.items())
                cf_acc = float(cf_scores.mean())
                mix_lat = sum(
                    w * float(np.mean([s["cost_lat"][a.split("_")[0].replace("slot", "CROP")]
                                       if a.startswith("CROP") else s["cost_lat"][a]
                                       for s in states]))
                    for a, w in weights.items() if a != "STOP")
                feas = [p for p in sweeps if p["vtok"] <= B + 1e-9]
                multi = max(feas, key=lambda p: p["acc"])
                fz = frozen_bt[round(frac, 2)]["multiaction"]["acc"]
                assert abs(multi["acc"] - fz) < 1e-9, \
                    f"{ds}/{scorer} B={frac}: recomputed multi {multi['acc']} != frozen {fz}"
                d = multi["_scores"] - mix_scores
                ci = hme.boot_ci(d)
                d_cf = multi["_scores"] - cf_scores
                ci_cf = hme.boot_ci(d_cf)
                wtxt = " + ".join(f"{w:.2f}·{a}" for a, w in
                                  sorted(weights.items(), key=lambda kv: -kv[1]))
                rows.append({"budget_frac": frac, "budget_vtok": B,
                             "mixture_weights": weights,
                             "mixture_acc": float(mix_acc),
                             "mixture_lat": float(mix_lat),
                             "mixture_cf_acc": cf_acc,
                             "mixture_cf_weights": cf_w,
                             "multiaction_acc": multi["acc"],
                             "multi_minus_mixture": float(d.mean()),
                             "ci95": ci,
                             "multi_minus_mixture_cf": float(d_cf.mean()),
                             "ci95_cf": ci_cf})
                md.append(f"| {frac:g}× | {wtxt} | {mix_acc:.3f} | "
                          f"{cf_acc:.3f} | {multi['acc']:.3f} | {d.mean():+.4f} "
                          f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {d_cf.mean():+.4f} "
                          f"[{ci_cf[0]:+.4f}, {ci_cf[1]:+.4f}] |")
            md.append("")
            out[f"{ds}/{scorer}"] = rows
    (ROOT / "paper/randomized_mixture_baseline.json").write_text(
        json.dumps(out, indent=1))
    (ROOT / "paper/randomized_mixture_baseline.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
