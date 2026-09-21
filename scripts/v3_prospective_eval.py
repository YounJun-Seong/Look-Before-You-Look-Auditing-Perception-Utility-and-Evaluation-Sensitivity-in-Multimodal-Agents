#!/usr/bin/env python3
"""Selector v3 prospective evaluation — single preregistered pass.

Analysis half of paper/PREREG_SELECTOR_V3.md on logs/v3_prospective.jsonl
(300 fresh states). Frozen components only: selector v3 (heads/scaler/
conf-ECDF/λ), selector v2 (for the head-to-head secondary), and the frozen
dev-fitted mixture weights (models/mixture_frozen_textvqa.json, applied at
this set's budgets). Strict primary; lenient rescore secondary.

Outputs paper/v3_prospective_eval.{json,md}.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppu.gate_features import extract  # noqa: E402

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "pe", ROOT / "scripts/prospective_eval.py")
pe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pe)

V3 = pickle.load(open(ROOT / "models/selector_frozen_v3.pkl", "rb"))
FAMILIES = ("CROP", "OCR", "HIGHRES")
FRACS = (0.25, 0.5, 1.0)


def v3_scores(states, Hmap, frac, scorer):
    H = np.stack([Hmap[st["sid"]] for st in states]).astype(np.float32)
    if V3["feature_set"] == "H":
        X = H
    else:  # H+F0
        srt = V3["conf_sorted_mean"]
        rank = lambda v: 1.0 - np.searchsorted(srt, v, side="right") / len(srt)
        F0 = np.array([extract({"answer": st["answer"], "question": st["question"],
                                "answer_len_tokens": st["alen"]},
                               float(rank(st["mean_logprob"]))) for st in states])
        X = np.hstack([H, F0])
    Xs = V3["scaler"].transform(X)
    dhat = {f: V3["heads"][f].predict(Xs) for f in FAMILIES}
    lam = V3["frozen_lambda_textvqa"][str(frac)]["lam"]
    out_s, out_c, mix = [], [], {}
    for i, st in enumerate(states):
        vals = {"STOP": 0.0}
        for f in FAMILIES:
            vals[f] = dhat[f][i] - lam * pe.fam_cost(st, f)
        fam = max(vals, key=vals.get)
        mix[fam] = mix.get(fam, 0) + 1
        out_s.append(float(np.mean(pe.fam_score(st, fam, scorer))))
        out_c.append(pe.fam_cost(st, fam))
    return (np.array(out_s), float(np.mean(out_c)),
            {k: round(v / len(states), 3) for k, v in sorted(mix.items())})


def main():
    states = pe.load_states("logs/v3_prospective.jsonl")
    pe.precompute(states)
    z = np.load(ROOT / "data/hidden/v3_prospective.npz", allow_pickle=True)
    Hmap = dict(zip(list(z["ids"]), z["H"]))
    missing = [st["sid"] for st in states if st["sid"] not in Hmap]
    assert not missing, f"missing embeddings: {len(missing)}"
    n = len(states)
    hi = float(np.mean([st["vtok"]["HIGHRES"] for st in states]))
    out = {"prereg": "paper/PREREG_SELECTOR_V3.md", "n_states": n,
           "highres_vtok": hi, "results": {}}
    md = [f"# Selector v3 prospective evaluation (preregistered, 1 pass; n={n})",
          "",
          f"always-HIGHRES {hi:.0f} tok. Frozen v3, frozen v2, frozen mixture;",
          "paired 10k bootstrap (seed 20260831).", ""]
    for scorer in ("strict", "lenient"):
        md += [f"## {scorer}", "",
               "| budget | v3 acc (vtok) | mixture | v2 | v3 − mixture "
               "[95% CI] | v3 − v2 [95% CI] |", "|---|---|--:|--:|:--|:--|"]
        res = {}
        for frac in FRACS:
            v3_s, v3_c, v3_mix = v3_scores(states, Hmap, frac, scorer)
            mix_s, mix_c = pe.mixture_scores(states, frac, scorer)
            v2_s, v2_c, _ = pe.selector_scores(states, frac, scorer)
            d_m, d_2 = v3_s - mix_s, v3_s - v2_s
            ci_m, ci_2 = pe.boot_ci(d_m), pe.boot_ci(d_2)
            res[str(frac)] = {
                "v3_acc": float(v3_s.mean()), "v3_vtok": v3_c, "v3_mix": v3_mix,
                "mixture_acc": float(mix_s.mean()), "v2_acc": float(v2_s.mean()),
                "v3_minus_mixture": float(d_m.mean()), "ci_mixture": ci_m,
                "v3_minus_v2": float(d_2.mean()), "ci_v2": ci_2}
            md.append(f"| {frac:g}× | {v3_s.mean():.3f} ({v3_c:.0f}) | "
                      f"{mix_s.mean():.3f} | {v2_s.mean():.3f} | "
                      f"{d_m.mean():+.4f} [{ci_m[0]:+.4f}, {ci_m[1]:+.4f}] | "
                      f"{d_2.mean():+.4f} [{ci_2[0]:+.4f}, {ci_2[1]:+.4f}] |")
        md += ["", "v3 action mix: " + json.dumps(
            {f: res[str(f)]["v3_mix"] for f in FRACS}), ""]
        out["results"][scorer] = res
    (ROOT / "paper/v3_prospective_eval.json").write_text(json.dumps(out, indent=1))
    (ROOT / "paper/v3_prospective_eval.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
