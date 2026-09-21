#!/usr/bin/env python3
"""Two-step sequential audit analysis (paper/PREREG_SEQUENTIAL.md, 1 pass).

Joins logs/seq_textvqa.jsonl (trajectories SEQ#0-3) with the one-step
references in logs/tv_200.jsonl on the 100-state subset and answers the four
preregistered questions. Strict scorer primary; lenient secondary. 10k
state-paired bootstrap (seed 20260831).

Known data limitation (recorded, not a protocol change): the Gate 0 one-step
arms were collected without logprobs, so the Q3 realizability signals are the
answer-level ones (did a1 change the STOP answer; a1 answer length); post-a1
token confidence is unavailable.

Outputs paper/sequential_audit.{json,md}.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppu.evaluator import normalize_answer  # noqa: E402

SEED = 20260831
N_BOOT = 10_000
CENTER = "0.2500,0.2500,0.7500,0.7500"
TOPRIGHT = "0.4500,0.0000,1.0000,0.5500"
A1S = ("center", "ocr")
TRAJ_OF = {("center", "crop_tr"): 0, ("center", "highres"): 1,
           ("ocr", "crop_tr"): 2, ("ocr", "highres"): 3}


def lenient(pred, golds):
    p = normalize_answer(pred or "")
    toks = p.split()

    def hit(g):
        g = normalize_answer(g)
        return g != "" and (g == p or (toks and g == toks[0]) or f" {g} " in f" {p} ")
    return min(1.0, sum(hit(g) for g in golds) / 3)


def load():
    seq = defaultdict(dict)   # sid -> (slot, rep) -> record
    for line in open(ROOT / "logs/seq_textvqa.jsonl"):
        r = json.loads(line)
        if r.get("type") == "rollout":
            seq[r["sample_id"]][(r["action"]["slot"], r["replay"])] = r
    ids = set(seq)
    ref = defaultdict(dict)   # sid -> (name, rep) -> record
    for line in open(ROOT / "logs/tv_200.jsonl"):
        r = json.loads(line)
        if r.get("type") != "rollout" or r["sample_id"] not in ids:
            continue
        a = r["action"]
        if a["type"] == "STOP":
            name = "stop"
        elif a["type"] == "OCR":
            name = "ocr"
        elif a["type"] == "HIGHRES":
            name = "highres"
        elif a["type"] == "CROP" and a.get("source") == "grid":
            b = ",".join(f"{v:.4f}" for v in a["box"])
            name = {CENTER: "center", TOPRIGHT: "crop_tr"}.get(b)
            if name is None:
                continue
        else:
            continue
        ref[r["sample_id"]][(name, r["replay"])] = r
    states = []
    for sid in sorted(ids):
        need = [(n, r) for n in ("stop", "center", "ocr", "crop_tr", "highres")
                for r in (1, 2)]
        if all(k in ref[sid] for k in need) and len(seq[sid]) == 8:
            states.append({"sid": sid, "seq": seq[sid], "ref": ref[sid],
                           "gold": ref[sid][("stop", 1)]["gold"]})
    return states


def sc(rec, st, scorer):
    if scorer == "strict":
        return float(rec["score_after"])
    return lenient(rec["answer_after"], st["gold"])


def boot_ci(d, seed=SEED):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    m = d[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def roc_auc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def main():
    states = load()
    n = len(states)
    out = {"prereg": "paper/PREREG_SEQUENTIAL.md", "n_states": n, "q": {}}
    md = [f"# Two-step sequential audit (preregistered, 1 pass; n={n})", ""]

    for scorer in ("strict", "lenient"):
        md += [f"## {scorer}", ""]
        q = {}

        # Q1: sequential opportunity per a1 — cross-replay oracle over
        # {stop-after-a1, a1->TR, a1->HIGHRES} minus one-step a1 value
        md += ["### Q1 — value of a second acquisition (cross-replay oracle "
               "over {stop-after-a1, a1→CROP-TR, a1→HIGHRES} − one-step a1)", "",
               "| a1 | one-step a1 | seq oracle | Δ [95% CI] |", "|---|--:|--:|:--|"]
        q1 = {}
        for a1 in A1S:
            base, gain = [], []
            for st in states:
                opts_by_rep = {}
                for rep in (1, 2):
                    opts_by_rep[rep] = {
                        "stay": sc(st["ref"][(a1, rep)], st, scorer),
                        "tr": sc(st["seq"][(TRAJ_OF[(a1, "crop_tr")], rep)], st, scorer),
                        "hi": sc(st["seq"][(TRAJ_OF[(a1, "highres")], rep)], st, scorer)}
                vals = []
                for sel, ev in ((1, 2), (2, 1)):
                    k = max(sorted(opts_by_rep[sel]), key=lambda k: (opts_by_rep[sel][k], k))
                    vals.append(opts_by_rep[ev][k])
                one = np.mean([opts_by_rep[r]["stay"] for r in (1, 2)])
                base.append(one)
                gain.append(np.mean(vals) - one)
            ci = boot_ci(gain)
            q1[a1] = {"one_step": float(np.mean(base)),
                      "seq_oracle_gain": float(np.mean(gain)), "ci95": ci}
            md.append(f"| {a1} | {np.mean(base):.3f} | "
                      f"{np.mean(base) + np.mean(gain):.3f} | "
                      f"{np.mean(gain):+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] |")
        q["q1"] = q1

        # Q2: history dependence — best a2 (incl. stopping) by what a1 revealed
        md += ["", "### Q2 — best follow-up by a1 outcome (R1-select, R2-score "
               "share of states; changed = a1 answer ≠ STOP answer)", "",
               "| a1 | condition | n | stay | →TR | →HIGHRES | mean continuation gain |",
               "|---|---|--:|--:|--:|--:|--:|"]
        q2 = []
        for a1 in A1S:
            groups = defaultdict(list)
            for st in states:
                ans_a1 = normalize_answer(st["ref"][(a1, 1)]["answer_after"] or "")
                ans_st = normalize_answer(st["ref"][("stop", 1)]["answer_after"] or "")
                changed = ans_a1 != ans_st
                correct = sc(st["ref"][(a1, 1)], st, scorer) >= 0.5
                groups[(changed, correct)].append(st)
            for (changed, correct), grp in sorted(groups.items()):
                best = Counter = defaultdict(int)
                gains = []
                for st in grp:
                    opts = {"stay": sc(st["ref"][(a1, 1)], st, scorer),
                            "tr": sc(st["seq"][(TRAJ_OF[(a1, "crop_tr")], 1)], st, scorer),
                            "hi": sc(st["seq"][(TRAJ_OF[(a1, "highres")], 1)], st, scorer)}
                    k = max(sorted(opts), key=lambda k: (opts[k], k))
                    ev = {"stay": sc(st["ref"][(a1, 2)], st, scorer),
                          "tr": sc(st["seq"][(TRAJ_OF[(a1, "crop_tr")], 2)], st, scorer),
                          "hi": sc(st["seq"][(TRAJ_OF[(a1, "highres")], 2)], st, scorer)}
                    best[k] += 1
                    gains.append(ev[k] - ev["stay"])
                m = len(grp)
                q2.append({"a1": a1, "changed": changed, "correct_a1": correct,
                           "n": m, "share": {k: best[k] / m for k in ("stay", "tr", "hi")},
                           "continuation_gain": float(np.mean(gains))})
                md.append(f"| {a1} | changed={int(changed)},a1-correct={int(correct)} "
                          f"| {m} | {best['stay']/m:.2f} | {best['tr']/m:.2f} "
                          f"| {best['hi']/m:.2f} | {np.mean(gains):+.3f} |")
        q["q2"] = q2

        # Q3: realizability — do post-a1 answer-level signals predict "a2 helps"?
        md += ["", "### Q3 — predicting 'a second acquisition helps' from "
               "post-a1 answer signals (labels cross-replay: R2)", "",
               "| a1 | P(a2 helps) | AUC: answer-changed | AUC: a1 answer len |",
               "|---|--:|--:|--:|"]
        q3 = {}
        for a1 in A1S:
            ys, chg, alen = [], [], []
            for st in states:
                stay2 = sc(st["ref"][(a1, 2)], st, scorer)
                cont2 = max(sc(st["seq"][(TRAJ_OF[(a1, "crop_tr")], 2)], st, scorer),
                            sc(st["seq"][(TRAJ_OF[(a1, "highres")], 2)], st, scorer))
                ys.append(int(cont2 > stay2 + 1e-9))
                a_ans = st["ref"][(a1, 1)]["answer_after"] or ""
                s_ans = st["ref"][("stop", 1)]["answer_after"] or ""
                chg.append(float(normalize_answer(a_ans) != normalize_answer(s_ans)))
                alen.append(float(len(a_ans)))
            q3[a1] = {"p_help": float(np.mean(ys)),
                      "auc_answer_changed": roc_auc(ys, chg),
                      "auc_answer_len": roc_auc(ys, alen)}
            f = lambda v: "—" if v is None else f"{v:.3f}"
            md.append(f"| {a1} | {np.mean(ys):.2f} | {f(q3[a1]['auc_answer_changed'])} "
                      f"| {f(q3[a1]['auc_answer_len'])} |")
        q["q3"] = q3

        # Q4: two-step vs one-step frontier points (no selection, mean replays)
        md += ["", "### Q4 — frontier points (mean over replays; extra vtok)", "",
               "| arm/trajectory | acc | vtok |", "|---|--:|--:|"]
        q4 = {}
        pts = [("STOP", lambda st: sc_pair(st, ("stop",), scorer), 0.0)]

        def sc_pair(st, path, scorer):
            if len(path) == 1:
                return np.mean([sc(st["ref"][(path[0], r)], st, scorer) for r in (1, 2)])
            slot = TRAJ_OF[path]
            return np.mean([sc(st["seq"][(slot, r)], st, scorer) for r in (1, 2)])
        for name in ("center", "ocr", "crop_tr", "highres"):
            acc = np.mean([sc_pair(st, (name,), scorer) for st in states])
            vt = np.mean([st["ref"][(name, 1)]["cost"]["visual_tokens"] for st in states])
            q4[name] = {"acc": float(acc), "vtok": float(vt)}
            md.append(f"| {name} (1-step) | {acc:.3f} | {vt:.0f} |")
        for path, slot in TRAJ_OF.items():
            acc = np.mean([sc_pair(st, path, scorer) for st in states])
            vt = np.mean([st["seq"][(slot, 1)]["cost"]["visual_tokens"] for st in states])
            q4["->".join(path)] = {"acc": float(acc), "vtok": float(vt)}
            md.append(f"| {path[0]}→{path[1]} | {acc:.3f} | {vt:.0f} |")
        md.append("")
        q["q4"] = q4
        out["q"][scorer] = q

    (ROOT / "paper/sequential_audit.json").write_text(json.dumps(out, indent=1))
    (ROOT / "paper/sequential_audit.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
