#!/usr/bin/env python3
"""Stronger-controller prospective evaluation (paper/PREREG_STRONG_CONTROLLER.md).

Policies on the fresh log (replays 1-2, frozen protocol):
  ridge / xgb / mlp   pool-trained frozen heads (models/strong_controllers_frozen.pkl)
  v2                  published dev-only frozen selector (reference)
  mixture             frozen dev-fitted state-independent mixture
  always-*            STOP / CROP(uniform grid) / OCR / HIGHRES
  random, best-fixed-at-B, Lagrangian oracle (cross-replay) for context
Budgets ¼/½/1× mean always-HIGHRES tokens; strict + lenient; state-paired
10k bootstrap for every contrast. Primary: xgb − mixture at ½× strict.
Co-secondary: xgb − ridge, mlp − ridge at ½× strict.
Usage: strong_controller_eval.py [--log logs/strong_textvqa.jsonl] [--tag strong]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".pylib"))
sys.path.insert(0, str(ROOT / "scripts"))
from strong_controller_common import (BUDGET_FRACS, FAMILIES, fam_cost,  # noqa: E402
                                      fam_score, features, load_states,
                                      make_rank, policy_from_dhat,
                                      precompute, scores)

SEED = 20260920
N_BOOT = 10_000
SC = pickle.load(open(ROOT / "models/strong_controllers_frozen.pkl", "rb"))
V2 = pickle.load(open(ROOT / "models/selector_frozen_v2.pkl", "rb"))
V2M = json.loads((ROOT / "models/selector_frozen_v2.json").read_text())
MIX = json.loads((ROOT / "models/mixture_frozen_textvqa.json").read_text())


def boot_ci(d, seed=SEED, n=N_BOOT):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def eval_fams(states, fams, scorer):
    acc = np.array([float(np.mean(fam_score(st, f, scorer))) for st, f in zip(states, fams)])
    cost = float(np.mean([fam_cost(st, f) for st, f in zip(states, fams)]))
    mix = defaultdict(int)
    for f in fams:
        mix[f] += 1
    return acc, cost, {k: round(v / len(fams), 3) for k, v in sorted(mix.items())}


def controller(states, cls, frac):
    rank = make_rank(SC["conf_sorted"])
    Xs = SC["scaler"].transform(features(states, rank))
    dhat = {f: SC["heads"][cls][f].predict(Xs) for f in FAMILIES}
    return policy_from_dhat(states, dhat, SC["lam"][cls][str(frac)])


def v2_policy(states, frac):
    rank = make_rank(V2["conf_sorted"]["mean"])
    Xs = V2["scaler"].transform(features(states, rank))
    dhat = {f: V2["heads"][f].predict(Xs) for f in FAMILIES}
    return policy_from_dhat(states, dhat, V2M["frozen_lambda_textvqa"][str(frac)]["lam"])


def mixture(states, frac, scorer):
    w = MIX["weights"][str(frac)]["w"]
    acc = np.array([sum(wt * float(np.mean(scores(st, a, scorer))) for a, wt in w.items())
                    for st in states])
    cost = float(np.mean([sum(wt * st["vtok"][a] for a, wt in w.items()) for st in states]))
    return acc, cost


def random_policy(states, B, scorer):
    """state-independent uniform-over-families randomization at cost B:
    mix always-STOP with the uniform family lottery (expectation)."""
    fam_acc = np.array([[float(np.mean(fam_score(st, f, scorer))) for f in ("STOP",) + FAMILIES]
                        for st in states])
    fam_cost_ = np.array([[fam_cost(st, f) for f in ("STOP",) + FAMILIES] for st in states])
    unif = fam_acc[:, 1:].mean(1); unif_c = fam_cost_[:, 1:].mean(1).mean()
    p = min(1.0, B / unif_c) if unif_c > 0 else 0.0
    return p * unif + (1 - p) * fam_acc[:, 0], p * unif_c


def best_fixed(states, B, scorer):
    best = None
    for f in ("STOP",) + FAMILIES:
        c = float(np.mean([fam_cost(st, f) for st in states]))
        a = float(np.mean([np.mean(fam_score(st, f, scorer)) for st in states]))
        if c <= B + 1e-9 and (best is None or a > best[0]):
            best = (a, f, c)
    return best


def lagrangian_oracle(states, B, scorer):
    keys = states[0]["keys"]
    assert all(st["keys"] == keys for st in states)
    Y = np.array([[scores(st, k, scorer) for k in keys] for st in states])
    C = np.array([[0.0 if k == "STOP" else st["vtok"][k] for k in keys] for st in states])
    best = None
    for lam in [0.0] + list(np.logspace(-7, -0.5, 121)):
        acc = cost = 0.0
        for sel, ev in ((0, 1), (1, 0)):
            idx = np.argmax(Y[:, :, sel] - lam * C, axis=1)
            acc += Y[np.arange(len(states)), idx, ev].mean()
            cost += C[np.arange(len(states)), idx].mean()
        acc, cost = acc / 2, cost / 2
        if cost <= B + 1e-9 and (best is None or acc > best):
            best = float(acc)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/strong_textvqa.jsonl")
    ap.add_argument("--tag", default="strong")
    args = ap.parse_args()
    states, dropped = load_states([args.log])
    assert dropped == 0, dropped
    precompute(states)
    n = len(states)
    hi_tok = float(np.mean([st["vtok"]["HIGHRES"] for st in states]))
    out = {"n": n, "highres_vtok": hi_tok, "log": args.log}
    md = [f"# Stronger-controller prospective evaluation ({args.tag}; n={n})", "",
          f"always-HIGHRES {hi_tok:.0f} tok. Pool-trained frozen ridge/xgb/mlp "
          f"(sha {json.loads((ROOT / 'models/strong_controllers_frozen.json').read_text())['sha256_16']}), "
          "frozen v2, frozen mixture; paired 10k bootstrap (seed 20260920).", ""]
    for scorer in ("strict", "lenient"):
        out[scorer] = {}
        always = {f: float(np.mean([np.mean(fam_score(st, f, scorer)) for st in states]))
                  for f in ("STOP",) + FAMILIES}
        md += [f"## {scorer}", "",
               " · ".join(f"always-{f} {v:.3f}" for f, v in always.items()), "",
               "| budget | ridge (vtok) | xgb (vtok) | mlp (vtok) | v2 (vtok) | mixture (vtok) | "
               "xgb − mix [CI] | xgb − ridge [CI] | mlp − ridge [CI] | ridge − mix [CI] | "
               "random | best-fixed | oracle |",
               "|---|---|---|---|---|---|:--|:--|:--|:--|--:|--:|--:|"]
        for frac in BUDGET_FRACS:
            B = frac * hi_tok
            pol, row = {}, {"budget_vtok": B}
            for cls in ("ridge", "xgb", "mlp"):
                a, c, mix = eval_fams(states, controller(states, cls, frac), scorer)
                pol[cls] = a; row[cls] = {"acc": float(a.mean()), "vtok": c, "mix": mix}
            a, c, mix = eval_fams(states, v2_policy(states, frac), scorer)
            pol["v2"] = a; row["v2"] = {"acc": float(a.mean()), "vtok": c, "mix": mix}
            pol["mix"], cm = mixture(states, frac, scorer)
            row["mixture"] = {"acc": float(pol["mix"].mean()), "vtok": cm}
            for name, (x, y) in {"xgb_minus_mix": ("xgb", "mix"), "xgb_minus_ridge": ("xgb", "ridge"),
                                 "mlp_minus_ridge": ("mlp", "ridge"), "ridge_minus_mix": ("ridge", "mix"),
                                 "mlp_minus_mix": ("mlp", "mix"), "xgb_minus_v2": ("xgb", "v2"),
                                 "ridge_minus_v2": ("ridge", "v2")}.items():
                d = pol[x] - pol[y]
                row[name] = {"delta": float(d.mean()), "ci": boot_ci(d)}
            ra, rc = random_policy(states, B, scorer)
            bf = best_fixed(states, B, scorer)
            row["random"] = {"acc": float(ra.mean()), "vtok": rc}
            row["best_fixed"] = {"acc": bf[0], "arm": bf[1], "vtok": bf[2]}
            row["oracle_lagrangian"] = lagrangian_oracle(states, B, scorer)
            out[scorer][str(frac)] = row
            f = lambda k: f"{row[k]['delta']:+.4f} [{row[k]['ci'][0]:+.4f}, {row[k]['ci'][1]:+.4f}]"
            md.append(f"| {frac}× | {row['ridge']['acc']:.3f} ({row['ridge']['vtok']:.0f}) | "
                      f"{row['xgb']['acc']:.3f} ({row['xgb']['vtok']:.0f}) | "
                      f"{row['mlp']['acc']:.3f} ({row['mlp']['vtok']:.0f}) | "
                      f"{row['v2']['acc']:.3f} ({row['v2']['vtok']:.0f}) | "
                      f"{row['mixture']['acc']:.3f} ({cm:.0f}) | {f('xgb_minus_mix')} | "
                      f"{f('xgb_minus_ridge')} | {f('mlp_minus_ridge')} | {f('ridge_minus_mix')} | "
                      f"{row['random']['acc']:.3f} | {bf[0]:.3f} | {row['oracle_lagrangian']:.3f} |")
        md += ["", "Action mixes:", ""]
        for frac in BUDGET_FRACS:
            r = out[scorer][str(frac)]
            md.append(f"- {frac}×: " + "; ".join(f"{cls} {r[cls]['mix']}" for cls in ("ridge", "xgb", "mlp", "v2")))
        md.append("")
    p = out["strict"]["0.5"]["xgb_minus_mix"]
    verdict = "PASS" if (p["ci"][0] > 0 or p["ci"][1] < 0) else "FAIL (CI includes 0)"
    md += [f"**Primary endpoint (xgb − mixture, ½× strict): {p['delta']:+.4f} "
           f"[{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}] → {verdict}**", ""]
    out["primary"] = {"contrast": "xgb_minus_mix @0.5x strict", **p, "verdict": verdict}
    (ROOT / f"paper/{args.tag}_controller_eval.json").write_text(json.dumps(out, indent=1))
    (ROOT / f"paper/{args.tag}_controller_eval.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
