#!/usr/bin/env python3
"""Phi-4 realizable policy check (post-hoc): Random < Confidence < Oracle.

The frozen v2.2 gate and multi-action value heads are Qwen-fitted and are not
transferred (backbone_check protocol). The realizable policy for the second
reader is therefore the zero-training confidence selector: look (HIGHRES) when
Phi-4's own STOP confidence (mean token logprob, from the byte-match-verified
side collection) is below a threshold, tuned only by budget.

Per dataset the table reports, at matched visual-token budgets, accuracy of
  random allocation  <=  confidence selector  <=  budget-constrained oracle
with a paired bootstrap CI on (confidence - random) at each budget.
Selection uses the same replay's confidence as the outcome being scored — a
runnable policy, no cross-replay peeking; the oracle column comes from
paper/p4_budget_frontier.json (cross-replay, Lagrangian sweep).

Usage:
    python scripts/p4_realizable.py \
        --pairs mme=logs/p4_mme200.jsonl:data/span_conf/p4_mme_stopconf.jsonl \
                textvqa=logs/p4_textvqa200.jsonl:data/span_conf/p4_textvqa_stopconf.jsonl \
        --frontier paper/p4_budget_frontier.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260830
N_BOOT = 10_000
BUDGET_FRACS = (0.05, 0.10, 0.20, 0.40, 1.00)  # matches budget_oracle_frontier


def load_pair(log, conf):
    rows = defaultdict(dict)  # sid -> per-replay dict
    for line in open(ROOT / log):
        r = json.loads(line)
        if r.get("type") != "rollout":
            continue
        t = r["action"]["type"]
        if t not in ("STOP", "HIGHRES"):
            continue
        d = rows[r["sample_id"]].setdefault(r["replay"], {})
        d[t] = r["score_after"]
        if t == "HIGHRES":
            d["vtok"] = r["cost"]["visual_tokens"]
    for line in open(ROOT / conf):
        r = json.loads(line)
        if r.get("type") != "span_conf":
            continue
        for rep in (1, 2):
            rr = r.get(f"r{rep}")
            if rr and rr.get("logprobs") and r["sample_id"] in rows \
                    and rep in rows[r["sample_id"]]:
                rows[r["sample_id"]][rep]["conf"] = float(np.mean(rr["logprobs"]))
    units = []
    for sid, reps in sorted(rows.items()):
        for rep, d in sorted(reps.items()):
            if all(k in d for k in ("STOP", "HIGHRES", "vtok", "conf")):
                units.append({"sid": sid, **d})
    return units


def curve(units, look_flags):
    acc = np.mean([u["HIGHRES"] if lk else u["STOP"]
                   for u, lk in zip(units, look_flags)])
    vtok = np.mean([u["vtok"] if lk else 0.0 for u, lk in zip(units, look_flags)])
    return acc, vtok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="name=p4log:confjsonl")
    ap.add_argument("--frontier", required=True)
    ap.add_argument("--out", default="p4_realizable")
    args = ap.parse_args()
    frontier = json.loads((ROOT / args.frontier).read_text())["datasets"]

    out = {}
    md = ["# Phi-4 realizable policy — Random vs confidence selector vs oracle",
          "",
          "Confidence = Phi-4's own STOP mean token logprob (verified side",
          "collection); selector looks (HIGHRES) below a budget-tuned threshold.",
          "Oracle = budget-constrained cross-replay oracle (all arms). Strict.", ""]
    for kv in args.pairs:
        name, rest = kv.split("=", 1)
        log, conf = rest.split(":", 1)
        units = load_pair(log, conf)
        confs = np.array([u["conf"] for u in units])
        hi_cost = float(np.mean([u["vtok"] for u in units]))
        rng = np.random.default_rng(SEED)

        taus = np.quantile(confs, np.linspace(0, 1, 81))
        conf_pts = []
        for t in taus:
            flags = confs < t
            a, v = curve(units, flags)
            conf_pts.append({"tau": float(t), "acc": a, "vtok": v,
                             "look": float(np.mean(flags))})
        stop_acc = float(np.mean([u["STOP"] for u in units]))
        hi_acc = float(np.mean([u["HIGHRES"] for u in units]))

        rows = []
        orc = {round(p["budget_frac"], 2): p
               for p in frontier[name]["budget_points"]}
        for f in BUDGET_FRACS:
            B = f * hi_cost
            p_look = min(1.0, B / hi_cost)
            rand_acc = (1 - p_look) * stop_acc + p_look * hi_acc
            feas = [p for p in conf_pts if p["vtok"] <= B + 1e-9]
            best = max(feas, key=lambda p: p["acc"])
            # paired bootstrap on (confidence - random) at this budget
            sel = np.array([u["HIGHRES"] if c < best["tau"] else u["STOP"]
                            for u, c in zip(units, confs)])
            rnd_mix = np.array([(1 - p_look) * u["STOP"] + p_look * u["HIGHRES"]
                                for u in units])
            d = sel - rnd_mix
            idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
            ci = [float(np.quantile(d[idx].mean(axis=1), q)) for q in (.025, .975)]
            o = orc.get(round(f, 2))
            rows.append({"budget_frac": f, "budget_vtok": B,
                         "random_acc": rand_acc, "conf_acc": best["acc"],
                         "conf_vtok": best["vtok"], "conf_look": best["look"],
                         "oracle_acc": o["acc"] if o else None,
                         "conf_minus_random": float(np.mean(d)), "ci95": ci})
        out[name] = {"n_units": len(units), "always_stop": stop_acc,
                     "always_highres": hi_acc, "highres_vtok": hi_cost,
                     "budget_rows": rows, "sweep": conf_pts}
        md += [f"## {name}  (units={len(units)}, STOP {stop_acc:.3f}, "
               f"HIGHRES {hi_acc:.3f} @ {hi_cost:.0f} tok)", "",
               "| budget | random | confidence | oracle | conf−random [CI] |",
               "|---|--:|--:|--:|:--|"]
        for r in rows:
            oa = "—" if r["oracle_acc"] is None else f"{r['oracle_acc']:.3f}"
            md.append(f"| {int(r['budget_frac']*100)}% | {r['random_acc']:.3f} "
                      f"| {r['conf_acc']:.3f} | {oa} "
                      f"| {r['conf_minus_random']:+.3f} "
                      f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] |")
        md.append("")
    (ROOT / f"paper/{args.out}.json").write_text(json.dumps(out, indent=1))
    (ROOT / f"paper/{args.out}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
