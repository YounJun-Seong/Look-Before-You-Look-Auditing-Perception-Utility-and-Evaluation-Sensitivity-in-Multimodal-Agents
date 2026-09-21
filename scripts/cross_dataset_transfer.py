#!/usr/bin/env python3
"""(vi) Cross-dataset frozen-selector transfer — zero inference, disclosed.

The selector v2 was trained on the Gate 0 dev pool and its per-budget lambda
was frozen on the TextVQA dev subset. Here it is applied WITHOUT ANY
ADAPTATION (same heads, same scaler, same conf-ECDF, same TextVQA lambda) to
the VQAv2 held-out arm set (never used for any selector training), against
the in-sample and cross-fitted state-independent mixtures of §8 and the
budget-constrained oracle. Budgets = frac x VQAv2 always-HIGHRES tokens.

Post-hoc, disclosed: the VQAv2 held-out set was analyzed before (frozen
multi-action v1 prereg eval); nothing here is fitted to it.

Outputs paper/cross_dataset_transfer.{json,md}.
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
spec.loader.exec_module(hme)
spec2 = importlib.util.spec_from_file_location(
    "rmb", ROOT / "scripts/randomized_mixture_baseline.py")
rmb = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(rmb)

from ppu.gate_features import extract  # noqa: E402

SEL = pickle.load(open(ROOT / "models/selector_frozen_v2.pkl", "rb"))
SELM = json.loads((ROOT / "models/selector_frozen_v2.json").read_text())
FAMILIES = ("CROP", "OCR", "HIGHRES")
FRACS = (0.25, 0.5, 1.0)


def selector_apply(states, frac, scorer):
    rank = lambda v, s=SEL["conf_sorted"]["mean"]: \
        1.0 - np.searchsorted(s, v, side="right") / len(s)
    X = np.array([extract({"answer": s["answer"], "question": s["question"],
                           "answer_len_tokens": s["answer_len_tokens"]},
                          float(rank(s["mean_logprob"]))) for s in states])
    Xs = SEL["scaler"].transform(X)
    dhat = {f: SEL["heads"][f].predict(Xs) for f in FAMILIES}
    lam = SELM["frozen_lambda_textvqa"][str(frac)]["lam"]  # transferred as-is
    scores, costs, mix = [], [], {}
    for i, s in enumerate(states):
        vals = {"STOP": 0.0}
        for f in FAMILIES:
            vals[f] = dhat[f][i] - lam * s["cost_vtok"][f]
        fam = max(vals, key=vals.get)
        mix[fam] = mix.get(fam, 0) + 1
        key = {"STOP": "stop", "CROP": "crop", "OCR": "ocr", "HIGHRES": "hi"}[fam]
        scores.append(float(np.mean(s[scorer][key])))
        costs.append(s["cost_vtok"][fam])
    return (np.array(scores), float(np.mean(costs)),
            {k: round(v / len(states), 3) for k, v in sorted(mix.items())})


def main():
    ds, fresh_log, arms_log = [x for x in hme.DS if x[0] == "vqav2"][0]
    states, _ = hme.load_states(ds, fresh_log, arms_log)
    hi = float(np.mean([s["cost_vtok"]["HIGHRES"] for s in states]))
    out = {"n_states": len(states), "highres_vtok": hi, "results": {}}
    md = ["# (vi) Cross-dataset frozen-selector transfer — TextVQA-frozen "
          "selector v2 on VQAv2 held-out (zero inference, disclosed post-hoc)",
          "",
          f"n = {len(states)}; no adaptation of heads, scaler, conf-ECDF or "
          "lambda (TextVQA values used verbatim). Mixtures per §8 machinery;",
          "paired 10k bootstrap.", ""]
    for scorer in ("strict", "lenient"):
        arms = rmb.arm_menu(states, scorer)
        md += [f"## {scorer}", "",
               "| budget | selector acc (vtok) | mix-CF acc | sel − mix-CF "
               "[95% CI] | sel − mix-insample [95% CI] | oracle (Lagr.) |",
               "|---|---|--:|:--|:--|--:|"]
        res = {}
        for frac in FRACS:
            B = frac * hi
            sel_s, sel_c, sel_mix = selector_apply(states, frac, scorer)
            w_in, _ = rmb.best_mixture(arms, B)
            mix_in = sum(w * arms[a][0] for a, w in w_in.items())
            idx = np.arange(len(states))
            mix_cf = np.zeros(len(states))
            for fold in (0, 1):
                tr = idx[idx % 2 == fold]
                te = idx[idx % 2 != fold]
                arms_tr = {a: (arms[a][0][tr], arms[a][1][tr]) for a in arms}
                w_tr, _ = rmb.best_mixture(arms_tr, B)
                mix_cf[te] = sum(w * arms[a][0][te] for a, w in w_tr.items())
            d_cf = sel_s - mix_cf
            d_in = sel_s - mix_in
            ci_cf, ci_in = hme.boot_ci(d_cf), hme.boot_ci(d_in)
            # Lagrangian oracle over deterministic per-state arms (cross-replay)
            pts = []
            for lam2 in [0.0] + list(np.logspace(-7, -0.5, 61)):
                accs, costs = [], []
                for sel_r, ev_r in ((0, 1), (1, 0)):
                    for s in states:
                        cand = [("stop", s[scorer]["stop"], 0.0),
                                ("ocr", s[scorer]["ocr"], s["cost_vtok"]["OCR"]),
                                ("hi", s[scorer]["hi"], s["cost_vtok"]["HIGHRES"]),
                                ("crop", s[scorer]["crop"], s["cost_vtok"]["CROP"])]
                        j = int(np.argmax([c[1][sel_r] - lam2 * c[2] for c in cand]))
                        accs.append(cand[j][1][ev_r])
                        costs.append(cand[j][2])
                pts.append((float(np.mean(costs)), float(np.mean(accs))))
            orc = max((a for c, a in pts if c <= B + 1e-9), default=None)
            res[str(frac)] = {"selector_acc": float(sel_s.mean()),
                              "selector_vtok": sel_c, "selector_mix": sel_mix,
                              "mix_cf": float(mix_cf.mean()),
                              "sel_minus_cf": float(d_cf.mean()), "ci_cf": ci_cf,
                              "sel_minus_in": float(d_in.mean()), "ci_in": ci_in,
                              "oracle": orc}
            md.append(f"| {frac:g}× | {sel_s.mean():.3f} ({sel_c:.0f}) | "
                      f"{mix_cf.mean():.3f} | {d_cf.mean():+.4f} "
                      f"[{ci_cf[0]:+.4f}, {ci_cf[1]:+.4f}] | {d_in.mean():+.4f} "
                      f"[{ci_in[0]:+.4f}, {ci_in[1]:+.4f}] | {orc:.3f} |")
        md += ["", "selector action mix: " + json.dumps(
            {f: res[str(f)]["selector_mix"] for f in FRACS}), ""]
        out["results"][scorer] = res
    (ROOT / "paper/cross_dataset_transfer.json").write_text(json.dumps(out, indent=1))
    (ROOT / "paper/cross_dataset_transfer.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
