#!/usr/bin/env python3
"""Budget-constrained cross-replay oracle frontier (post-hoc, no new inference).

Computes, on the Gate 0 brute-force logs, the oracle

    max_g  E[y(g(s))]   s.t.   E[c(g(s))] <= B

via its Lagrangian sweep: for each lambda, g_lambda(s) = argmax_a
y_sel(a) - lambda * c(a) chosen on the selection replay, scored on the other
replay, symmetrised over both directions. Reading off the best sweep point
with mean extra visual tokens <= B traces the concave frontier. Budgets are
B in {5, 10, 20, 40, 100}% of the dataset's always-HIGHRES token cost.

Action set: STOP + the deterministic arms only (grid CROPs, OCR, HIGHRES),
matched across replays by action key; VLM crop proposals are excluded (uneven
coverage, 70% parse failure in Gate 0), consistent with the selector analysis.

Outputs paper/budget_oracle_frontier.{json,md}.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGS = {"mme": "logs/g0_mme200.jsonl", "textvqa": "logs/tv_200.jsonl"}
OUT = "budget_oracle_frontier"
BUDGET_FRACS = (0.05, 0.10, 0.20, 0.40, 1.00)
LAMBDAS = [0.0] + list(np.logspace(-7, -0.5, 121))


def deterministic(a) -> bool:
    if a["type"] == "STOP":
        return True
    if a["type"] == "CROP" and a.get("source") == "grid":
        return True
    return a["type"] in ("OCR", "HIGHRES")


def load(log):
    # sample -> replay -> action_key -> (score, vtok)
    data = defaultdict(lambda: defaultdict(dict))
    for line in open(ROOT / log):
        r = json.loads(line)
        if r.get("type") != "rollout":
            continue
        a = r["action"]
        if not deterministic(a):
            continue
        key = "STOP" if a["type"] == "STOP" else a.get("key") or a["type"]
        data[r["sample_id"]][r["replay"]][key] = (
            r["score_after"], r["cost"]["visual_tokens"])
    # keep only samples with both replays and identical key sets
    out = {}
    for sid, reps in data.items():
        if 1 in reps and 2 in reps:
            keys = sorted(set(reps[1]) & set(reps[2]))
            if "STOP" in keys:
                out[sid] = {1: {k: reps[1][k] for k in keys},
                            2: {k: reps[2][k] for k in keys}}
    return out


def sweep(data):
    pts = []
    sids = sorted(data)
    for lam in LAMBDAS:
        accs, costs = [], []
        for sel, ev in ((1, 2), (2, 1)):
            for sid in sids:
                dsel, dev = data[sid][sel], data[sid][ev]
                k = max(dsel, key=lambda k: (dsel[k][0] - lam * dsel[k][1],
                                             -dsel[k][1], k))
                accs.append(dev[k][0])
                costs.append(dsel[k][1])
        pts.append({"lambda": lam, "acc": float(np.mean(accs)),
                    "vtok": float(np.mean(costs))})
    return pts


def main():
    global LOGS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*", default=[],
                    help="name=path pairs; default Gate 0 Qwen logs")
    ap.add_argument("--out", default="budget_oracle_frontier",
                    help="output basename under paper/")
    args = ap.parse_args()
    if args.logs:
        LOGS = dict(kv.split("=", 1) for kv in args.logs)
    OUT = args.out
    out = {"lambdas": len(LAMBDAS), "budget_fracs": BUDGET_FRACS, "datasets": {}}
    md = ["# Budget-constrained cross-replay oracle frontier (post-hoc, Gate 0)",
          "",
          "max_g E[y(g(s))] s.t. E[c(g(s))] <= B, computed by Lagrangian sweep",
          "(select on one replay, score on the other, symmetrised). Deterministic",
          "arms only (STOP, grid CROPs, OCR, HIGHRES); strict scores as logged.",
          ""]
    for ds, log in LOGS.items():
        data = load(log)
        pts = sweep(data)
        stop = float(np.mean([np.mean([data[s][r]["STOP"][0] for r in (1, 2)])
                              for s in data]))
        hr_keys = [k for k in next(iter(data.values()))[1] if k.startswith("HIGHRES")]
        hi_cost = float(np.mean([data[s][1][hr_keys[0]][1] for s in data])) if hr_keys else None
        unconstrained = max(pts, key=lambda p: p["acc"])
        rows = []
        for f in BUDGET_FRACS:
            B = f * hi_cost
            feas = [p for p in pts if p["vtok"] <= B + 1e-9]
            best = max(feas, key=lambda p: p["acc"])
            rec = (best["acc"] - stop) / (unconstrained["acc"] - stop)
            rows.append({"budget_frac": f, "budget_vtok": B, **best,
                         "recovery_of_oracle_gain": rec})
        out["datasets"][ds] = {
            "n_states": len(data), "always_stop": stop,
            "always_highres_vtok": hi_cost,
            "oracle_unconstrained": unconstrained,
            "budget_points": rows, "sweep": pts,
        }
        md += [f"## {ds}  (n={len(data)}, STOP {stop:.3f}, "
               f"unconstrained oracle {unconstrained['acc']:.3f} "
               f"@ {unconstrained['vtok']:.0f} tok; always-HIGHRES cost "
               f"{hi_cost:.0f} tok)", "",
               "| budget | tok | oracle acc | Δ vs STOP | % of oracle gain |",
               "|---|--:|--:|--:|--:|"]
        for r in rows:
            md.append(f"| {int(r['budget_frac']*100)}% | {r['budget_vtok']:.0f} "
                      f"| {r['acc']:.3f} | +{r['acc']-stop:.3f} "
                      f"| {100*r['recovery_of_oracle_gain']:.0f}% |")
        md.append("")
    (ROOT / f"paper/{OUT}.json").write_text(json.dumps(out, indent=1))
    (ROOT / f"paper/{OUT}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
