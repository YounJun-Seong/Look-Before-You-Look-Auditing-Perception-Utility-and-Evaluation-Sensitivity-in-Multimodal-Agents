#!/usr/bin/env python3
"""Hidden × nonlinear controller prospective evaluation (paper/PREREG_HIDDEN_CONTROLLER.md).

All nine frozen (features × head) cells + frozen v2 + frozen mixture on the
fresh log; budgets ¼/½/1× mean always-HIGHRES tokens; strict + lenient;
state-paired 10k bootstrap. Primary: (H/primary-nonlinear) − mixture at
½× strict. Key contrasts: H/xgb − F0/xgb (representation), H/xgb − H/ridge
(capacity), H+F0/xgb − mixture.
Usage: hidden_controller_eval.py [--log logs/hidden_textvqa.jsonl]
       [--hidden data/hidden/hidden_textvqa.npz] [--tag hidden]
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

SEED = 20260921
N_BOOT = 10_000
HC = pickle.load(open(ROOT / "models/hidden_controllers_frozen.pkl", "rb"))
HCM = json.loads((ROOT / "models/hidden_controllers_frozen.json").read_text())
V2 = pickle.load(open(ROOT / "models/selector_frozen_v2.pkl", "rb"))
V2M = json.loads((ROOT / "models/selector_frozen_v2.json").read_text())
MIX = json.loads((ROOT / "models/mixture_frozen_textvqa.json").read_text())
CELLS = [f"{fs}/{cls}" for fs in ("F0", "H", "H+F0") for cls in ("ridge", "xgb", "mlp")]


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


def build_X(states, fs, Hmap):
    F0 = features(states, make_rank(HC["conf_sorted"]))
    if fs == "F0":
        return F0
    H = np.stack([Hmap[s["sid"]] for s in states])
    return H if fs == "H" else np.hstack([H, F0])


def cell_policy(states, cell, frac, Hmap):
    fs, _ = cell.split("/")
    Xz = HC["scalers"][cell].transform(build_X(states, fs, Hmap))
    dhat = {f: HC["heads"][cell][f].predict(Xz) for f in FAMILIES}
    return policy_from_dhat(states, dhat, HC["lam"][cell][str(frac)])


def v2_policy(states, frac):
    Xz = V2["scaler"].transform(features(states, make_rank(V2["conf_sorted"]["mean"])))
    dhat = {f: V2["heads"][f].predict(Xz) for f in FAMILIES}
    return policy_from_dhat(states, dhat, V2M["frozen_lambda_textvqa"][str(frac)]["lam"])


def mixture(states, frac, scorer):
    w = MIX["weights"][str(frac)]["w"]
    acc = np.array([sum(wt * float(np.mean(scores(st, a, scorer))) for a, wt in w.items()) for st in states])
    cost = float(np.mean([sum(wt * st["vtok"][a] for a, wt in w.items()) for st in states]))
    return acc, cost


def lagrangian_oracle(states, B, scorer):
    keys = states[0]["keys"]; assert all(st["keys"] == keys for st in states)
    Y = np.array([[scores(st, k, scorer) for k in keys] for st in states])
    Cc = np.array([[0.0 if k == "STOP" else st["vtok"][k] for k in keys] for st in states])
    best = None
    for lam in [0.0] + list(np.logspace(-7, -0.5, 121)):
        acc = cost = 0.0
        for sel, ev in ((0, 1), (1, 0)):
            idx = np.argmax(Y[:, :, sel] - lam * Cc, axis=1)
            acc += Y[np.arange(len(states)), idx, ev].mean(); cost += Cc[np.arange(len(states)), idx].mean()
        acc, cost = acc / 2, cost / 2
        if cost <= B + 1e-9 and (best is None or acc > best):
            best = float(acc)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/hidden_textvqa.jsonl")
    ap.add_argument("--hidden", default="data/hidden/hidden_textvqa.npz")
    ap.add_argument("--tag", default="hidden")
    args = ap.parse_args()
    states, dropped = load_states([args.log]); assert dropped == 0, dropped
    precompute(states)
    z = np.load(ROOT / args.hidden, allow_pickle=True)
    Hmap = {str(s): v.astype(np.float32) for s, v in zip(z["ids"], z["H"])}
    assert all(st["sid"] in Hmap for st in states), "hidden states missing for some eval states"
    n = len(states)
    hi_tok = float(np.mean([st["vtok"]["HIGHRES"] for st in states]))
    primary_cell = HC["primary"]
    out = {"n": n, "highres_vtok": hi_tok, "log": args.log, "primary_cell": primary_cell}
    md = [f"# Hidden × nonlinear controller prospective evaluation ({args.tag}; n={n})", "",
          f"always-HIGHRES {hi_tok:.0f} tok. Nine frozen cells (sha {HCM['sha256_16']}, pool n={HC['n_train']}), "
          f"frozen v2, frozen mixture; paired 10k bootstrap (seed {SEED}). Primary cell: **{primary_cell}**.", ""]
    contrasts = [("primary_minus_mix", primary_cell, "mix"), ("H_xgb_minus_mix", "H/xgb", "mix"),
                 ("H_mlp_minus_mix", "H/mlp", "mix"), ("HF0_xgb_minus_mix", "H+F0/xgb", "mix"),
                 ("H_ridge_minus_mix", "H/ridge", "mix"), ("F0_xgb_minus_mix", "F0/xgb", "mix"),
                 ("F0_ridge_minus_mix", "F0/ridge", "mix"),
                 ("H_xgb_minus_F0_xgb", "H/xgb", "F0/xgb"), ("H_xgb_minus_H_ridge", "H/xgb", "H/ridge"),
                 ("H_mlp_minus_H_ridge", "H/mlp", "H/ridge"), ("HF0_xgb_minus_F0_xgb", "H+F0/xgb", "F0/xgb"),
                 ("H_ridge_minus_F0_ridge", "H/ridge", "F0/ridge")]
    for scorer in ("strict", "lenient"):
        out[scorer] = {}
        always = {f: float(np.mean([np.mean(fam_score(st, f, scorer)) for st in states])) for f in ("STOP",) + FAMILIES}
        md += [f"## {scorer}", "", " · ".join(f"always-{f} {v:.3f}" for f, v in always.items()), "",
               "Accuracy (realized vtok) per cell:", "",
               "| budget | " + " | ".join(CELLS) + " | v2 | mixture | oracle |",
               "|---|" + "---|" * (len(CELLS) + 3)]
        rows = {}
        for frac in BUDGET_FRACS:
            B = frac * hi_tok; pol, row = {}, {"budget_vtok": B}
            for cell in CELLS:
                a, c, mix = eval_fams(states, cell_policy(states, cell, frac, Hmap), scorer)
                pol[cell] = a; row[cell] = {"acc": float(a.mean()), "vtok": c, "mix": mix}
            a, c, mix = eval_fams(states, v2_policy(states, frac), scorer)
            pol["v2"] = a; row["v2"] = {"acc": float(a.mean()), "vtok": c, "mix": mix}
            pol["mix"], cm = mixture(states, frac, scorer); row["mixture"] = {"acc": float(pol["mix"].mean()), "vtok": cm}
            row["oracle_lagrangian"] = lagrangian_oracle(states, B, scorer)
            for name, x, y in contrasts:
                d = pol[x] - pol[y]; row[name] = {"delta": float(d.mean()), "ci": boot_ci(d)}
            out[scorer][str(frac)] = row; rows[frac] = row
            md.append(f"| {frac}× | " + " | ".join(f"{row[c]['acc']:.3f} ({row[c]['vtok']:.0f})" for c in CELLS)
                      + f" | {row['v2']['acc']:.3f} ({row['v2']['vtok']:.0f}) | {row['mixture']['acc']:.3f} ({cm:.0f}) | {row['oracle_lagrangian']:.3f} |")
        md += ["", "Contrasts (Δ accuracy [95% CI]):", "", "| contrast | ¼× | ½× | 1× |", "|---|:--|:--|:--|"]
        for name, x, y in contrasts:
            md.append(f"| {x} − {y} | " + " | ".join(
                f"{rows[fr][name]['delta']:+.4f} [{rows[fr][name]['ci'][0]:+.4f}, {rows[fr][name]['ci'][1]:+.4f}]"
                for fr in BUDGET_FRACS) + " |")
        md += ["", "Action mixes (½×): " + "; ".join(f"{c} {rows[0.5][c]['mix']}" for c in CELLS), ""]
    p = out["strict"]["0.5"]["primary_minus_mix"]
    verdict = "PASS" if (p["ci"][0] > 0 or p["ci"][1] < 0) else "FAIL (CI includes 0)"
    md += [f"**Primary endpoint ({primary_cell} − mixture, ½× strict): {p['delta']:+.4f} "
           f"[{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}] → {verdict}**", ""]
    out["primary"] = {"contrast": f"{primary_cell} − mixture @0.5x strict", **p, "verdict": verdict}
    (ROOT / f"paper/{args.tag}_controller_eval.json").write_text(json.dumps(out, indent=1))
    (ROOT / f"paper/{args.tag}_controller_eval.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
