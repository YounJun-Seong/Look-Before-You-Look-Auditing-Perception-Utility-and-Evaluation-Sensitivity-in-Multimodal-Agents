#!/usr/bin/env python3
"""(A) Realized-cost matched mixture (post-hoc, zero inference, disclosed).

The prospective table compares the frozen selector against the mixture at the
NOMINAL budgets; the selector's realized token spend is lower (53/129/212 vs
77/154/241). This closes the "same realized cost?" objection: the dev-fitted
mixture is re-solved at exactly the selector's realized mean token cost
(weights still fitted on Gate 0 dev arm means only — deployability preserved)
and compared paired on the same 600 prospective states, strict and lenient.

Outputs paper/costmatched_mixture.{json,md}.
"""
from __future__ import annotations

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "pe", ROOT / "scripts/prospective_eval.py")
pe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pe)

PROS = json.loads((ROOT / "paper/prospective_eval.json").read_text())


def dev_arm_means(log="logs/tv_200.jsonl"):
    per = defaultdict(lambda: defaultdict(list))
    for line in open(ROOT / log):
        r = json.loads(line)
        if r.get("type") != "rollout":
            continue
        a, sid = r["action"], r["sample_id"]
        if a["type"] == "STOP":
            name = "STOP"
        elif a["type"] in ("OCR", "HIGHRES"):
            name = a["type"]
        elif a["type"] == "CROP" and a.get("source") == "grid":
            name = f"CROP@{pe.box_key(a['box'])}"
        else:
            continue
        per[name][sid].append((r["score_after"], r["cost"]["visual_tokens"]))
    return {name: (float(np.mean([np.mean([x[0] for x in v]) for v in d.values()])),
                   float(np.mean([np.mean([x[1] for x in v]) for v in d.values()])))
            for name, d in per.items()}


def best_mixture(menu, B):
    names = sorted(menu)
    best = None
    for a in names:
        if menu[a][1] <= B + 1e-9 and (best is None or menu[a][0] > best[1]):
            best = ({a: 1.0}, menu[a][0])
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            lo, hi = sorted((a, b), key=lambda x: menu[x][1])
            if not (menu[lo][1] <= B + 1e-9 < menu[hi][1]):
                continue
            w = (B - menu[lo][1]) / (menu[hi][1] - menu[lo][1])
            v = (1 - w) * menu[lo][0] + w * menu[hi][0]
            if v > best[1]:
                best = ({lo: 1 - w, hi: w}, v)
    return best


def main():
    states = pe.load_states("logs/prospective_textvqa.jsonl")
    pe.precompute(states)
    menu = dev_arm_means()
    out = {}
    md = ["# (A) Realized-cost matched mixture vs frozen selector (post-hoc)",
          "",
          "Dev-fitted mixture re-solved at the selector's REALIZED mean token",
          "cost per budget (weights from Gate 0 dev arm means only). Paired",
          "10k bootstrap on the 600 prospective states.", ""]
    for scorer in ("strict", "lenient"):
        md += [f"## {scorer}", "",
               "| budget | matched B (tok) | mixture (weights) | mixture acc "
               "(vtok) | selector acc (vtok) | sel − mix [95% CI] |",
               "|---|--:|---|---|---|:--|"]
        res = {}
        for frac in pe.FRACS:
            prow = PROS["results"][scorer][str(frac)]
            B = prow["selector_vtok"]
            weights, _ = best_mixture(menu, B)
            mix_s, mix_c = [], []
            for st in states:
                s = sum(w * float(np.mean(pe.scores(st, a, scorer)))
                        for a, w in weights.items())
                c = sum(w * (0.0 if a == "STOP" else st["vtok"][a])
                        for a, w in weights.items())
                mix_s.append(s)
                mix_c.append(c)
            mix_s = np.array(mix_s)
            sel_s, sel_c, _ = pe.selector_scores(states, frac, scorer)
            d = sel_s - mix_s
            ci = pe.boot_ci(d)
            wtxt = " + ".join(f"{w:.2f}·{a.replace('CROP@','C@')}" for a, w in
                              sorted(weights.items(), key=lambda kv: -kv[1]))
            res[str(frac)] = {"matched_B": B, "weights": weights,
                              "mixture_acc": float(mix_s.mean()),
                              "mixture_vtok": float(np.mean(mix_c)),
                              "selector_acc": float(sel_s.mean()),
                              "selector_vtok": sel_c,
                              "sel_minus_mix": float(d.mean()), "ci95": ci}
            md.append(f"| {frac:g}× | {B:.0f} | {wtxt} | {mix_s.mean():.3f} "
                      f"({np.mean(mix_c):.0f}) | {sel_s.mean():.3f} ({sel_c:.0f}) "
                      f"| {d.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
        md.append("")
        out[scorer] = res
    (ROOT / "paper/costmatched_mixture.json").write_text(json.dumps(out, indent=1))
    (ROOT / "paper/costmatched_mixture.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
