#!/usr/bin/env python3
"""Per-action-family damage decomposition (post-hoc, no new inference).

Addresses the "you have ten additional actions, of course at least one hurts"
critique of the aggregate DMG number: decomposes harm/help per action family
on the Gate 0 brute-force logs. For each state x replay with a logged STOP:

  harm  = additional action's score < STOP score   (given STOP correct)
  help  = additional action's score > STOP score   (given STOP wrong)

reported both as `any` (>=1 action in the family, the DMG-style existential)
and `single` (uniform-random action in the family — the deployable rate).
`net` = mean score(a) - score(STOP) over all states, uniform within family.
Costs are mean extra visual tokens per family.

Outputs paper/per_action_harm.{json,md}.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LOGS = {"mme": "logs/g0_mme200.jsonl", "textvqa": "logs/tv_200.jsonl"}
OUT = "per_action_harm"

# family key -> predicate over action dict
FAMS = {
    "CROP_grid": lambda a: a["type"] == "CROP" and a.get("source") == "grid",
    "CROP_question": lambda a: a["type"] == "CROP" and a.get("source") == "vlm_proposal",
    "CROP_random": lambda a: a["type"] == "CROP_RANDOM",
    "OCR": lambda a: a["type"] == "OCR",
    "HIGHRES": lambda a: a["type"] == "HIGHRES",
}
# the aggregate-DMG action set (matches the paper's "additional actions")
ANY_SET = ("CROP_grid", "CROP_question", "OCR", "HIGHRES")


def fam_of(a):
    for k, pred in FAMS.items():
        if pred(a):
            return k
    return None


def main():
    global LOGS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*", default=[],
                    help="name=path pairs; default Gate 0 Qwen logs")
    ap.add_argument("--out", default=OUT, help="output basename under paper/")
    args = ap.parse_args()
    if args.logs:
        LOGS = dict(kv.split("=", 1) for kv in args.logs)
    OUT = args.out
    out = {}
    for ds, log in LOGS.items():
        # (sample, replay) -> {"STOP": score, fam: [(score, vtok), ...]}
        st = defaultdict(lambda: defaultdict(list))
        stop = {}
        for line in open(ROOT / log):
            r = json.loads(line)
            if r.get("type") != "rollout":
                continue
            key = (r["sample_id"], r["replay"])
            if r["action"]["type"] == "STOP":
                stop[key] = r["score_after"]
                continue
            f = fam_of(r["action"])
            if f:
                st[key][f].append((r["score_after"], r["cost"]["visual_tokens"]))

        rows = {}
        units = [k for k in stop if k in st]
        for f in list(FAMS) + ["any_additional"]:
            harm_any = harm_single = n_correct = 0.0
            help_any = help_single = n_wrong = 0.0
            net = net_n = cost = cost_n = 0.0
            covered = 0
            for key in units:
                s0 = stop[key]
                if f == "any_additional":
                    acts = [x for g in ANY_SET for x in st[key].get(g, [])]
                else:
                    acts = st[key].get(f, [])
                if not acts:
                    continue
                covered += 1
                scores = [x[0] for x in acts]
                net += sum(scores) / len(scores) - s0
                net_n += 1
                cost += sum(x[1] for x in acts) / len(acts)
                cost_n += 1
                if s0 >= 1.0:
                    n_correct += 1
                    harm_any += any(s < s0 for s in scores)
                    harm_single += sum(s < s0 for s in scores) / len(scores)
                elif s0 <= 0.0:
                    n_wrong += 1
                    help_any += any(s > s0 for s in scores)
                    help_single += sum(s > s0 for s in scores) / len(scores)
            rows[f] = {
                "dmg_unconditional": harm_any / len(units) if units else None,
                "coverage": covered, "units": len(units),
                "harm_any_given_stop_correct": harm_any / n_correct if n_correct else None,
                "harm_single_given_stop_correct": harm_single / n_correct if n_correct else None,
                "n_stop_correct": int(n_correct),
                "help_any_given_stop_wrong": help_any / n_wrong if n_wrong else None,
                "help_single_given_stop_wrong": help_single / n_wrong if n_wrong else None,
                "n_stop_wrong": int(n_wrong),
                "net_improvement": net / net_n if net_n else None,
                "mean_visual_tokens": cost / cost_n if cost_n else None,
            }
        out[ds] = rows

    (ROOT / f"paper/{OUT}.json").write_text(json.dumps(out, indent=1))

    md = ["# Per-action damage decomposition (post-hoc, Gate 0 logs)", "",
          "Unit = state x replay with a logged STOP. `any` = >=1 action in the",
          "family flips (the DMG-style existential); `single` = uniform-random",
          "single action in the family (deployable rate). Scores strict as logged.",
          "Note: unit here is state x replay; the paper's pooled DMG (13.8/13.0)",
          "uses per-direction pooling in ppu/audit.py — textvqa reproduces exactly,",
          "mme differs by 2.5pp under the unit definition (both are reported).",
          ""]
    for ds, rows in out.items():
        md += [f"## {ds}", "",
               "| family | n(units) | DMG (uncond.) | P(harm\\|STOP✓) any / single | "
               "P(help\\|STOP✗) any / single | net Δscore | vtok |",
               "|---|--:|--:|:--|:--|--:|--:|"]
        for f, r in rows.items():
            fmt = lambda v: "—" if v is None else f"{v:.3f}"
            vt = r["mean_visual_tokens"]
            md.append(
                f"| {f} | {r['coverage']} | {fmt(r['dmg_unconditional'])} | "
                f"{fmt(r['harm_any_given_stop_correct'])}"
                f" / {fmt(r['harm_single_given_stop_correct'])} | "
                f"{fmt(r['help_any_given_stop_wrong'])} / "
                f"{fmt(r['help_single_given_stop_wrong'])} | "
                f"{fmt(r['net_improvement'])} | "
                f"{'—' if vt is None else f'{vt:.0f}'} |")
        md.append("")
    (ROOT / f"paper/{OUT}.md").write_text("\n".join(md))
    print(json.dumps({ds: {f: rows[f]["harm_any_given_stop_correct"]
                           for f in rows} for ds, rows in out.items()}, indent=1))


if __name__ == "__main__":
    main()
