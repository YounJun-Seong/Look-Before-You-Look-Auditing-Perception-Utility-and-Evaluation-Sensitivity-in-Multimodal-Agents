#!/usr/bin/env python3
"""Deployable-action restricted oracle — robustness check of the structural
oracle (zero inference; 2026-09-21).

The structural audit's oracle selects among the individual arms
{STOP, CROP#0..7, OCR, HIGHRES}; the deployable selectors choose a FAMILY
and treat CROP as the uniform-grid family expectation. The action spaces
differ, so part of the oracle–policy gap could be crop granularity. Here the
oracle is recomputed on exactly the deployable decision space

    A_i^dep = {STOP, CROP-family, OCR, HIGHRES},

with CROP-family outcome = mean over the 8 grid crops (per replay) and cost =
mean grid-crop tokens — identical to what every selector used — under the
SAME protocol as the audit: select on replay 1 (ties → cheaper family, then
key; ppu/audit.py::_argmax_by_cost), score on replay 2; best fixed policy =
10× random 2-fold cross-fitted 'always X' over the 11 fixed arms
(ppu/audit.py::best_fixed_cf, seed 20260816); headroom H = oracle − best
fixed with 10k state-paired bootstrap. Both oracles are compared against the
SAME best-fixed baseline so the difference is purely the oracle's action
space. Budget-constrained (Lagrangian, cross-replay symmetrised) versions at
¼/½/1× always-HIGHRES are reported alongside. Deterministic arms only
(no VLM proposals, no FULL, no dummies).
Outputs paper/deployable_oracle.{json,md}.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ppu.schema import COST_RANK  # noqa: E402
from strong_controller_common import lenient  # noqa: E402

SEED = 20260816
N_BOOT = 10_000
FRACS = (0.25, 0.5, 1.0)
DEP = ("STOP", "CROPFAM", "OCR", "HIGHRES")
SETS = [
    ("textvqa / Gate-0 (Qwen, n=200)", ["logs/tv_200.jsonl"], True),
    ("mme / Gate-0 (Qwen, n=196)", ["logs/g0_mme200.jsonl"], False),
    ("textvqa / held-out 250", ["logs/fresh_textvqa_orig.jsonl", "logs/heldout_textvqa_arms.jsonl"], True),
    ("textvqa / prospective 600", ["logs/prospective_textvqa.jsonl"], True),
    ("textvqa / v3-prospective 300", ["logs/v3_prospective.jsonl"], True),
    ("textvqa / strong 400", ["logs/strong_textvqa.jsonl"], True),
    ("textvqa / hidden 400", ["logs/hidden_textvqa.jsonl"], True),
    ("textvqa / Gate-0 Phi-4 (n=200)", ["logs/p4_textvqa200.jsonl"], True),
    ("mme / Gate-0 Phi-4 (n=196)", ["logs/p4_mme200.jsonl"], False),
]


def fam_of(key):
    return "CROP" if key.startswith("CROP") else key.split("#")[0]


def load(logs):
    by = defaultdict(lambda: {"s": defaultdict(dict), "l": defaultdict(dict), "v": {}})
    for log in logs:
        for line in open(ROOT / log):
            r = json.loads(line)
            if r.get("type") != "rollout" or r.get("dummy") or r.get("error"):
                continue
            a, rep = r["action"], r["replay"]
            if rep not in (1, 2):
                continue
            if a["type"] == "STOP":
                key = "STOP"
            elif a["type"] in ("OCR", "HIGHRES"):
                key = a["type"]
            elif a["type"] == "CROP" and a.get("source") == "grid":
                key = f"CROP#{a['slot']}"
            else:
                continue
            st = by[r["sample_id"]]
            st["s"][key][rep] = float(r["score_after"])
            st["l"][key][rep] = lenient(r["answer_after"], r["gold"])
            st["v"][key] = float(r["cost"]["visual_tokens"])
    states = []
    for sid, st in sorted(by.items()):
        keys = sorted(k for k in st["s"] if set(st["s"][k]) >= {1, 2})
        crops = [k for k in keys if k.startswith("CROP#")]
        if not ({"STOP", "OCR", "HIGHRES"} <= set(keys) and crops):
            continue
        for sc in ("s", "l"):
            for rep in (1, 2):
                st[sc]["CROPFAM"][rep] = float(np.mean([st[sc][k][rep] for k in crops]))
        st["v"]["CROPFAM"] = float(np.mean([st["v"][k] for k in crops]))
        st["v"]["STOP"] = 0.0
        st["sid"], st["keys"], st["crops"] = sid, keys, crops
        states.append(st)
    return states


def argmax_by_cost(keys, scores):
    return min(keys, key=lambda k: (-scores[k], COST_RANK.get(fam_of(k), 9), k))


def oracle_xr(states, arms_fn, sc, sel=1, ev=2):
    out = []
    for st in states:
        keys = arms_fn(st)
        k = argmax_by_cost(keys, {a: st[sc][a][sel] for a in keys})
        out.append(st[sc][k][ev])
    return np.array(out)


def best_fixed_cf(states, policies, sc):
    """ppu/audit.py::best_fixed_cf rule: 10 random 2-fold splits, pick by mean
    replay-1 score on train (ties → cheaper), score replay 2 on test."""
    rng = np.random.default_rng(SEED)
    acc = {st["sid"]: [] for st in states}
    chosen = Counter()
    for _ in range(10):
        perm = rng.permutation(len(states))
        folds = [[states[i] for i in perm[0::2]], [states[i] for i in perm[1::2]]]
        for i in (0, 1):
            train, test = folds[i], folds[1 - i]
            pick = max(policies, key=lambda p: (
                float(np.mean([v[sc][p][1] if p in v[sc] else 0.0 for v in train])),
                -COST_RANK.get(fam_of(p), 9)))
            chosen[pick] += 1
            for v in test:
                acc[v["sid"]].append(v[sc][pick][2] if pick in v[sc] else 0.0)
    per = np.array([float(np.mean(acc[st["sid"]])) for st in states])
    return per, ",".join(f"{p}:{c}" for p, c in chosen.most_common(2))


def boot(d):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    m = d[idx].mean(axis=1)
    return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def lagrangian(states, arms_fn, sc, B):
    keys = sorted(set().union(*(set(arms_fn(st)) for st in states)))
    # arms absent for a state (fewer grid slots) get -inf on selection, 0 on evaluation
    Y = np.array([[[st[sc][k][r] if k in st[sc] else 0.0 for r in (1, 2)] for k in keys] for st in states])
    M = np.array([[0.0 if k in st[sc] else -np.inf for k in keys] for st in states])
    C = np.array([[st["v"].get(k, 0.0) for k in keys] for st in states])
    best = None
    for lam in [0.0] + list(np.logspace(-7, -0.5, 121)):
        acc = cost = 0.0
        for sel, ev in ((0, 1), (1, 0)):
            idx = np.argmax(Y[:, :, sel] - lam * C + M, axis=1)
            acc += Y[np.arange(len(states)), idx, ev].mean()
            cost += C[np.arange(len(states)), idx].mean()
        acc, cost = acc / 2, cost / 2
        if cost <= B + 1e-9 and (best is None or acc > best):
            best = float(acc)
    return best


def fmt(v, ci=None, pp=True):
    s = f"{100*v:+.1f}" if pp else f"{v:.3f}"
    return s if ci is None else f"{s} [{100*ci[0]:+.1f}, {100*ci[1]:+.1f}]"


def main():
    out, md = {}, ["# Deployable-action restricted oracle (robustness check of the structural oracle)", "",
                   "full = oracle over the individual arms {STOP, CROP#0..7, OCR, HIGHRES}; dep = oracle over the",
                   "deployable decision space {STOP, CROP-family (uniform-grid expectation), OCR, HIGHRES}. Both:",
                   "select on replay 1 (ties → cheaper family), score on replay 2; headroom H = oracle − the SAME",
                   "cross-fitted best fixed arm (audit rule, 11 fixed arms). 10k state-paired bootstrap CIs. pp.", ""]
    full_arms = lambda st: ["STOP", "OCR", "HIGHRES"] + st["crops"]
    dep_arms = lambda st: list(DEP)
    for name, logs, has_len in SETS:
        states = load(logs)
        n = len(states)
        hi = float(np.mean([st["v"]["HIGHRES"] for st in states]))
        res = {"n": n, "highres_vtok": hi}
        md += [f"## {name}  (n={n}, always-HIGHRES {hi:.0f} tok)", "",
               "| scorer | best fixed | full oracle | dep oracle | H_full [CI] | H_dep [CI] | full − dep [CI] | dep/full | sym: H_full / H_dep |",
               "|---|--:|--:|--:|:--|:--|:--|--:|:--|"]
        for sc, lab in (("s", "strict"), ("l", "lenient")):
            if sc == "l" and not has_len:
                continue
            bf, picks = best_fixed_cf(states, ["STOP", "OCR", "HIGHRES"] + [f"CROP#{i}" for i in range(8)], sc)
            of = oracle_xr(states, full_arms, sc); od = oracle_xr(states, dep_arms, sc)
            of_s = (of + oracle_xr(states, full_arms, sc, 2, 1)) / 2
            od_s = (od + oracle_xr(states, dep_arms, sc, 2, 1)) / 2
            bfd, _ = best_fixed_cf(states, list(DEP), sc)
            row = {"best_fixed": float(bf.mean()), "best_fixed_picks": picks,
                   "best_fixed_dep_space": float(bfd.mean()),
                   "oracle_full": float(of.mean()), "oracle_dep": float(od.mean()),
                   "H_full": float((of - bf).mean()), "H_full_ci": boot(of - bf),
                   "H_dep": float((od - bf).mean()), "H_dep_ci": boot(od - bf),
                   "full_minus_dep": float((of - od).mean()), "full_minus_dep_ci": boot(of - od),
                   "H_dep_over_H_full": float((od - bf).mean() / (of - bf).mean()) if (of - bf).mean() else None,
                   "sym_H_full": float((of_s - bf).mean()), "sym_H_dep": float((od_s - bf).mean()),
                   "budget": {}}
            for frac in FRACS:
                B = frac * hi
                row["budget"][str(frac)] = {"full": lagrangian(states, full_arms, sc, B),
                                            "dep": lagrangian(states, dep_arms, sc, B)}
            res[lab] = row
            md.append(f"| {lab} | {bf.mean():.3f} | {of.mean():.3f} | {od.mean():.3f} | "
                      f"{fmt(row['H_full'], row['H_full_ci'])} | {fmt(row['H_dep'], row['H_dep_ci'])} | "
                      f"{fmt(row['full_minus_dep'], row['full_minus_dep_ci'])} | "
                      f"{row['H_dep_over_H_full']:.2f} | {fmt(row['sym_H_full'])} / {fmt(row['sym_H_dep'])} |")
        md += ["", "Budget-constrained oracle accuracy (Lagrangian, cross-replay symmetrised), full / dep:", "",
               "| scorer | ¼× | ½× | 1× |", "|---|:--|:--|:--|"]
        for lab in ("strict", "lenient"):
            if lab in res:
                md.append(f"| {lab} | " + " | ".join(
                    f"{res[lab]['budget'][str(f)]['full']:.3f} / {res[lab]['budget'][str(f)]['dep']:.3f}"
                    for f in FRACS) + " |")
        md.append("")
        out[name] = res
        print(f"{name}: H_full {100*res['strict']['H_full']:+.1f} H_dep {100*res['strict']['H_dep']:+.1f} "
              f"(ratio {res['strict']['H_dep_over_H_full']:.2f})", flush=True)
    (ROOT / "paper/deployable_oracle.json").write_text(json.dumps(out, indent=1))
    (ROOT / "paper/deployable_oracle.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
