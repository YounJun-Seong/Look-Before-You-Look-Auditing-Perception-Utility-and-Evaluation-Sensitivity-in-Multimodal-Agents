#!/usr/bin/env python3
"""Cost-aware sequential controller v2 on the 300-state sequential set.

DISCLOSED SECONDARY analysis: the evaluation set (logs/seq_prospective.jsonl)
was already analyzed for the v1 controller prereg; nothing here is fitted to
it (heads/λ frozen on the dev-100, sha in models/seq_controller_v2_frozen.json).
Confidence features come from the logprobs recorded at collection time.

Outputs paper/seq_v2_eval.{json,md}.
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "spe", ROOT / "scripts/seq_prospective_eval.py")
spe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spe)

V2 = pickle.load(open(ROOT / "models/seq_controller_v2_frozen.pkl", "rb"))
V1 = pickle.load(open(ROOT / "models/seq_controller_frozen.pkl", "rb"))
OCR_CUE = re.compile(r"\b(say|says|written|word|text|sign|read|letter|number|"
                     r"brand|name|title|label)\b", re.I)
A1S = ("center", "ocr")


def rows_for(states, a1, scorer):
    out = []
    for st in states:
        for rep in (1, 2):
            stop_rec = st["ref"][("stop", rep)]
            a1_rec = st["ref"][(a1, rep)]
            pre = (stop_rec.get("confidence") or {}).get("mean_logprob")
            post = (a1_rec.get("confidence") or {}).get("mean_logprob")
            pmin = (a1_rec.get("confidence") or {}).get("min_logprob")
            if pre is None or post is None:
                continue
            a_ans = a1_rec["answer_after"] or ""
            s_ans = stop_rec["answer_after"] or ""
            x = [float(spe.normalize_answer(a_ans) != spe.normalize_answer(s_ans)),
                 float(len(a_ans)), float(pre), float(post),
                 float(post - pre), float(pmin),
                 1.0 if OCR_CUE.search(stop_rec["question"] or "") else 0.0]
            y = {"stay": spe.sc(a1_rec, st, scorer),
                 "tr": spe.sc(st["seq"][(spe.TRAJ_OF[(a1, "tr")], rep)], st, scorer),
                 "hi": spe.sc(st["seq"][(spe.TRAJ_OF[(a1, "hi")], rep)], st, scorer)}
            a1c = a1_rec["cost"]["visual_tokens"]
            c = {"stay": 0.0,
                 "tr": st["seq"][(spe.TRAJ_OF[(a1, "tr")], rep)]["cost"]["visual_tokens"] - a1c,
                 "hi": st["seq"][(spe.TRAJ_OF[(a1, "hi")], rep)]["cost"]["visual_tokens"] - a1c}
            out.append({"x": x, "y": y, "c": c})
    return out


def v2_choices(rows, a1, lam):
    m = V2["heads"][a1]
    mu, sd = np.array(m["mu"]), np.array(m["sd"])
    X = (np.array([r["x"] for r in rows]) - mu) / sd
    A = np.hstack([X, np.ones((len(X), 1))])
    d = {o: A @ np.array(m["w"][o]) for o in ("tr", "hi")}
    out = []
    for j, r in enumerate(rows):
        vals = {"stay": 0.0, "tr": d["tr"][j] - lam * r["c"]["tr"],
                "hi": d["hi"][j] - lam * r["c"]["hi"]}
        out.append(max(vals, key=vals.get))
    return out


def v1_choices(rows, a1):
    heads = V1["heads"]["strict"][a1]
    out = []
    for r in rows:
        A = np.array([r["x"][0], r["x"][1], 1.0])  # v1 features: changed, len
        vals = {"stay": 0.0,
                "tr": float(A @ np.array(heads["tr"])),
                "hi": float(A @ np.array(heads["hi"]))}
        out.append(max(vals, key=vals.get))
    return out


def main():
    states = spe.load()
    out = {"n_states": len(states), "results": {}}
    md = [f"# Cost-aware sequential controller v2 on the 300-state set "
          f"(disclosed secondary; n={len(states)})", "",
          "Frozen on dev-100 (sha in models/seq_controller_v2_frozen.json;",
          "a1-confidence regeneration byte-match 379/400, disclosed). Paired",
          "10k bootstrap vs the frozen mixture (= always→TR).", ""]
    for scorer in ("strict", "lenient"):
        md += [f"## {scorer}", "",
               "| a1 | policy | acc | Δvtok | look | vs always→TR [95% CI] |",
               "|---|---|--:|--:|--:|:--|"]
        res = {}
        for a1 in A1S:
            rows = rows_for(states, a1, scorer)
            tr = np.array([r["y"]["tr"] for r in rows])
            stay = np.array([r["y"]["stay"] for r in rows])
            entries = {"stop_after_a1": (stay, 0.0, 0.0),
                       "always_tr(mixture)": (tr, float(np.mean(
                           [r["c"]["tr"] for r in rows])), 1.0)}
            for name, ch in (("v1_controller", v1_choices(rows, a1)),
                             ("v2_lam0", v2_choices(rows, a1, 0.0)),
                             ("v2_lamB", v2_choices(rows, a1,
                                                    V2["lambda"][a1]["lamB"]))):
                acc = np.array([r["y"][c] for r, c in zip(rows, ch)])
                cost = float(np.mean([r["c"][c] for r, c in zip(rows, ch)]))
                look = float(np.mean([c != "stay" for c in ch]))
                entries[name] = (acc, cost, look)
            res[a1] = {}
            for name, (acc, cost, look) in entries.items():
                d = acc - tr
                ci = spe.boot_ci(d)
                res[a1][name] = {"acc": float(acc.mean()), "cost": cost,
                                 "look": look, "vs_tr": float(d.mean()),
                                 "ci95": ci}
                md.append(f"| {a1} | {name} | {acc.mean():.3f} | {cost:.0f} | "
                          f"{look:.2f} | {d.mean():+.4f} "
                          f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |")
        md.append("")
        out["results"][scorer] = res
    (ROOT / "paper/seq_v2_eval.json").write_text(json.dumps(out, indent=1))
    (ROOT / "paper/seq_v2_eval.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
