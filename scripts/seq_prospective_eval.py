#!/usr/bin/env python3
"""Sequential prospective evaluation — single preregistered pass.

Analysis half of paper/PREREG_SEQ_PROSPECTIVE.md on logs/seq_prospective.jsonl
(self-contained: one-step arms + SEQ trajectories in one log). The frozen
controller heads and the frozen mixture (= always->TR on both branches) come
from models/seq_controller_frozen.pkl; nothing is fitted here.

Outputs paper/seq_prospective_eval.{json,md}.
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppu.evaluator import normalize_answer  # noqa: E402

SEED = 20260831
N_BOOT = 10_000
FROZEN = pickle.load(open(ROOT / "models/seq_controller_frozen.pkl", "rb"))
A1S = ("center", "ocr")
CENTER = "0.2500,0.2500,0.7500,0.7500"
TOPRIGHT = "0.4500,0.0000,1.0000,0.5500"
TRAJ_OF = {("center", "tr"): 0, ("center", "hi"): 1,
           ("ocr", "tr"): 2, ("ocr", "hi"): 3}


def lenient(pred, golds):
    p = normalize_answer(pred or "")
    toks = p.split()

    def hit(g):
        g = normalize_answer(g)
        return g != "" and (g == p or (toks and g == toks[0]) or f" {g} " in f" {p} ")
    return min(1.0, sum(hit(g) for g in golds) / 3)


def load():
    ref = defaultdict(dict)
    seq = defaultdict(dict)
    gold = {}
    for line in open(ROOT / "logs/seq_prospective.jsonl"):
        r = json.loads(line)
        if r.get("type") != "rollout":
            continue
        sid, a, rep = r["sample_id"], r["action"], r["replay"]
        gold[sid] = r["gold"]
        if a["type"] == "SEQ":
            seq[sid][(a["slot"], rep)] = r
        elif a["type"] == "STOP":
            ref[sid][("stop", rep)] = r
        elif a["type"] == "OCR":
            ref[sid][("ocr", rep)] = r
        elif a["type"] == "HIGHRES":
            ref[sid][("highres", rep)] = r
        elif a["type"] == "CROP":
            b = ",".join(f"{v:.4f}" for v in a["box"])
            name = {CENTER: "center", TOPRIGHT: "crop_tr"}[b]
            ref[sid][(name, rep)] = r
    states = []
    for sid in sorted(gold):
        if len(seq[sid]) == 8 and len(ref[sid]) == 10:
            states.append({"sid": sid, "ref": ref[sid], "seq": seq[sid],
                           "gold": gold[sid]})
    return states


def sc(rec, st, scorer):
    if scorer == "strict":
        return float(rec["score_after"])
    return lenient(rec["answer_after"], st["gold"])


def boot_ci(d):
    d = np.asarray(d, float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    m = d[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def branch_rows(states, a1, scorer):
    rows = []
    for st in states:
        for rep in (1, 2):
            a_ans = st["ref"][(a1 if a1 != "center" else "center", rep)]["answer_after"] or ""
            s_ans = st["ref"][("stop", rep)]["answer_after"] or ""
            x = [float(normalize_answer(a_ans) != normalize_answer(s_ans)),
                 float(len(a_ans))]
            y = {"stay": sc(st["ref"][(a1, rep)], st, scorer),
                 "tr": sc(st["seq"][(TRAJ_OF[(a1, "tr")], rep)], st, scorer),
                 "hi": sc(st["seq"][(TRAJ_OF[(a1, "hi")], rep)], st, scorer)}
            a1c = st["ref"][(a1, rep)]["cost"]["visual_tokens"]
            c = {"stay": 0.0,
                 "tr": st["seq"][(TRAJ_OF[(a1, "tr")], rep)]["cost"]["visual_tokens"] - a1c,
                 "hi": st["seq"][(TRAJ_OF[(a1, "hi")], rep)]["cost"]["visual_tokens"] - a1c}
            rows.append({"x": x, "y": y, "c": c})
    return rows


def ctrl_choice(rows, heads):
    out = []
    for r in rows:
        A = np.array(r["x"] + [1.0])
        vals = {"stay": 0.0,
                "tr": float(A @ np.array(heads["tr"])),
                "hi": float(A @ np.array(heads["hi"]))}
        out.append(max(vals, key=vals.get))
    return out


def main():
    states = load()
    n = len(states)
    out = {"prereg": "paper/PREREG_SEQ_PROSPECTIVE.md", "n_states": n,
           "results": {}}
    md = [f"# Sequential prospective evaluation (preregistered, 1 pass; n={n})",
          "",
          "Frozen controller (strict-trained heads primary; lenient-trained",
          "variant disclosed) vs frozen mixture (= always→CROP-TR). Same-replay",
          "decision+outcome, mean over replays; paired 10k bootstrap.", ""]
    for scorer in ("strict", "lenient"):
        md += [f"## {scorer} scorer", ""]
        res = {}
        for a1 in A1S:
            rows = branch_rows(states, a1, scorer)
            stay = np.array([r["y"]["stay"] for r in rows])
            tr = np.array([r["y"]["tr"] for r in rows])
            hi = np.array([r["y"]["hi"] for r in rows])
            branch = {"stop_after_a1": float(stay.mean()),
                      "always_tr_mixture": float(tr.mean()),
                      "always_hi": float(hi.mean()),
                      "oracle_crossreplay": None}
            # cross-replay oracle ceiling
            orc = []
            for st in states:
                o = {}
                for rep in (1, 2):
                    o[rep] = {"stay": sc(st["ref"][(a1, rep)], st, scorer),
                              "tr": sc(st["seq"][(TRAJ_OF[(a1, "tr")], rep)], st, scorer),
                              "hi": sc(st["seq"][(TRAJ_OF[(a1, "hi")], rep)], st, scorer)}
                vals = []
                for sel, ev in ((1, 2), (2, 1)):
                    k = max(sorted(o[sel]), key=lambda k: (o[sel][k], k))
                    vals.append(o[ev][k])
                orc.append(np.mean(vals))
            branch["oracle_crossreplay"] = float(np.mean(orc))
            for variant in ("strict", "lenient"):
                heads = FROZEN["heads"][variant][a1]
                ch = ctrl_choice(rows, heads)
                acc = np.array([r["y"][c] for r, c in zip(rows, ch)])
                cost = float(np.mean([r["c"][c] for r, c in zip(rows, ch)]))
                look = float(np.mean([c != "stay" for c in ch]))
                d_mix = acc - tr
                d_stay = acc - stay
                branch[f"controller_{variant}heads"] = {
                    "acc": float(acc.mean()), "cost": cost, "look": look,
                    "minus_mixture": float(d_mix.mean()), "ci95_mix": boot_ci(d_mix),
                    "minus_stay": float(d_stay.mean()), "ci95_stay": boot_ci(d_stay)}
            res[a1] = branch
            b = branch
            cs_, cl_ = b["controller_strictheads"], b["controller_lenientheads"]
            md += [f"### a1 = {a1}", "",
                   "| policy | acc |", "|---|--:|",
                   f"| stop after a1 | {b['stop_after_a1']:.3f} |",
                   f"| frozen mixture (= always→TR) | {b['always_tr_mixture']:.3f} |",
                   f"| always→HIGHRES | {b['always_hi']:.3f} |",
                   f"| controller (strict-heads) | {cs_['acc']:.3f} (look {cs_['look']:.2f}) |",
                   f"| controller (lenient-heads) | {cl_['acc']:.3f} (look {cl_['look']:.2f}) |",
                   f"| sequential oracle (cross-replay) | {b['oracle_crossreplay']:.3f} |",
                   "",
                   f"strict-heads − mixture = {cs_['minus_mixture']:+.4f} "
                   f"[{cs_['ci95_mix'][0]:+.4f}, {cs_['ci95_mix'][1]:+.4f}]; "
                   f"lenient-heads − mixture = {cl_['minus_mixture']:+.4f} "
                   f"[{cl_['ci95_mix'][0]:+.4f}, {cl_['ci95_mix'][1]:+.4f}]", ""]
        out["results"][scorer] = res
    (ROOT / "paper/seq_prospective_eval.json").write_text(json.dumps(out, indent=1))
    (ROOT / "paper/seq_prospective_eval.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
