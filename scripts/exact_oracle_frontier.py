#!/usr/bin/env python3
"""Exact budget-constrained oracle via multiple-choice knapsack DP (post-hoc).

Closes the "the frontier is only a Lagrangian lower bound" loophole. For each
total budget C = frac x n x mean(always-HIGHRES tokens), solve EXACTLY

    max_{a_1..a_n} sum_i y_i^{sel}(a_i)   s.t.   sum_i c_i(a_i) <= C

by DP over integer token costs (selection objective on one replay), then score
the chosen assignment on the other replay; symmetrised over both directions —
the same cross-replay protocol as the Lagrangian sweep, with the relaxation
removed. Also reports the randomized-policy value at C (upper concave
envelope over the Lagrangian sweep's evaluated points).

Same data and deterministic arm set as scripts/budget_oracle_frontier.py.
Outputs paper/exact_oracle_frontier.{json,md}.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "bof", ROOT / "scripts/budget_oracle_frontier.py")
bof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bof)

BUDGET_FRACS = (0.05, 0.10, 0.20, 0.40, 1.00)


def mckp(states, C):
    """states: list of (ys, cs) arrays per state. Returns per-state chosen arm
    indices for the exact optimum with total cost <= C."""
    C = int(C)
    NEG = -1e18
    dp = np.full(C + 1, NEG)
    dp[0] = 0.0
    choice = np.zeros((len(states), C + 1), dtype=np.int8)
    for i, (ys, cs) in enumerate(states):
        ndp = np.full(C + 1, NEG)
        pick = np.zeros(C + 1, dtype=np.int8)
        for j, (y, c) in enumerate(zip(ys, cs)):
            c = int(c)
            if c > C:
                continue
            cand = np.full(C + 1, NEG)
            cand[c:] = dp[: C + 1 - c] + y
            upd = cand > ndp
            ndp[upd] = cand[upd]
            pick[upd] = j
        dp = ndp
        choice[i] = pick
    b = int(np.argmax(dp))
    idx = np.zeros(len(states), dtype=np.int8)
    for i in range(len(states) - 1, -1, -1):
        j = choice[i, b]
        idx[i] = j
        b -= int(states[i][1][j])
    return idx, dp.max()


def hull_value(pts, B):
    """randomized-policy value at mean-budget B: best chord over evaluated
    Lagrangian points (vtok, acc)."""
    pts = sorted({(round(p["vtok"], 6), p["acc"]) for p in pts})
    best = max((a for v, a in pts if v <= B + 1e-9), default=None)
    for i, (v1, a1) in enumerate(pts):
        for v2, a2 in pts[i + 1:]:
            if v1 <= B + 1e-9 < v2:
                w = (B - v1) / (v2 - v1)
                val = (1 - w) * a1 + w * a2
                if val > best:
                    best = val
    return best


def main():
    out = {}
    md = ["# Exact budget-constrained oracle (MCKP DP) vs Lagrangian sweep", "",
          "Same cross-replay protocol and deterministic arms as the Lagrangian",
          "frontier; the DP removes the relaxation on the selection objective.",
          "`randomized` = upper concave envelope of the evaluated Lagrangian",
          "points (a state-independent randomization over sweep operating",
          "points). Strict scores as logged.", ""]
    for ds, log in bof.LOGS.items():
        data = bof.load(log)
        sids = sorted(data)
        n = len(sids)
        lag = bof.sweep(data)
        stop = float(np.mean([np.mean([data[s][r]["STOP"][0] for r in (1, 2)])
                              for s in data]))
        hr = [k for k in data[sids[0]][1] if k.startswith("HIGHRES")]
        hi = float(np.mean([data[s][1][hr[0]][1] for s in data]))
        rows = []
        md += [f"## {ds}  (n={n}, STOP {stop:.3f}, always-HIGHRES {hi:.0f} tok)", "",
               "| budget | Lagrangian acc | exact DP acc | randomized hull | "
               "DP − Lagrangian |", "|---|--:|--:|--:|--:|"]
        for f in BUDGET_FRACS:
            B = f * hi
            feas = [p for p in lag if p["vtok"] <= B + 1e-9]
            lag_acc = max(p["acc"] for p in feas)
            accs = []
            for sel, ev in ((1, 2), (2, 1)):
                sts = []
                keys = {}
                for s in sids:
                    ks = sorted(data[s][sel])
                    keys[s] = ks
                    sts.append((np.array([data[s][sel][k][0] for k in ks]),
                                np.array([data[s][sel][k][1] for k in ks])))
                idx, _ = mckp(sts, B * n)
                accs.append(np.mean([data[s][ev][keys[s][j]][0]
                                     for s, j in zip(sids, idx)]))
            dp_acc = float(np.mean(accs))
            rnd = hull_value(lag, B)
            rows.append({"budget_frac": f, "budget_vtok": B,
                         "lagrangian_acc": lag_acc, "exact_dp_acc": dp_acc,
                         "randomized_hull_acc": rnd,
                         "dp_minus_lagrangian": dp_acc - lag_acc})
            md.append(f"| {int(f*100)}% | {lag_acc:.3f} | {dp_acc:.3f} | "
                      f"{rnd:.3f} | {dp_acc - lag_acc:+.4f} |")
        md.append("")
        out[ds] = rows
    (ROOT / "paper/exact_oracle_frontier.json").write_text(json.dumps(out, indent=1))
    (ROOT / "paper/exact_oracle_frontier.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
