#!/usr/bin/env python3
"""(B) History-conditioned two-step controller + hierarchy, (C) transfer.

Dev role: the 100-state sequential set. After a1 the controller sees ONLY
post-a1 observables — [a1 answer != STOP answer, len(a1 answer)] — and picks
{stay, ->CROP-TR, ->HIGHRES} via two ridge Δ-heads (Δ(a2) = y(a1->a2) -
y(stay)); decision = argmax(0, Δ̂_TR, Δ̂_HI). Deliberately minimal: no
budget term, no tuning beyond a fixed alpha=1.0, features fixed a priori.

Hierarchy (per a1, same-replay decision+outcome, mean over replays; the
sequential cross-replay oracle from the audit is the ceiling):
  stop-after-a1 / always->TR / always->HIGHRES / random continuation at the
  controller's rate / state-independent {stay,TR,HI} mixture at the
  controller's realized incremental cost (in-sample AND cross-fitted) /
  controller (cross-fitted) / sequential oracle (cross-replay).

(C) cross-action transfer: heads fitted on one a1 branch applied unchanged
to the other; compared against the natively cross-fitted controller.

Outputs paper/seq_controller.{json,md}.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "sa", ROOT / "scripts/sequential_analysis.py")
sa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sa)

SEED = 20260831
OPTS = ("stay", "tr", "hi")


def ridge_fit(X, y, alpha=1.0):
    A = np.hstack([X, np.ones((len(X), 1))])
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y)
    return w


def ridge_pred(w, X):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def branch_data(states, a1, scorer):
    """per state x replay: features + option scores + incremental costs."""
    rows = []
    for st in states:
        for rep in (1, 2):
            a_ans = st["ref"][(a1, rep)]["answer_after"] or ""
            s_ans = st["ref"][("stop", rep)]["answer_after"] or ""
            feats = [float(sa.normalize_answer(a_ans) != sa.normalize_answer(s_ans)),
                     float(len(a_ans))]
            y = {"stay": sa.sc(st["ref"][(a1, rep)], st, scorer),
                 "tr": sa.sc(st["seq"][(sa.TRAJ_OF[(a1, "crop_tr")], rep)], st, scorer),
                 "hi": sa.sc(st["seq"][(sa.TRAJ_OF[(a1, "highres")], rep)], st, scorer)}
            a1_cost = st["ref"][(a1, rep)]["cost"]["visual_tokens"]
            c = {"stay": 0.0,
                 "tr": st["seq"][(sa.TRAJ_OF[(a1, "crop_tr")], rep)]["cost"]["visual_tokens"] - a1_cost,
                 "hi": st["seq"][(sa.TRAJ_OF[(a1, "highres")], rep)]["cost"]["visual_tokens"] - a1_cost}
            rows.append({"i": len(rows), "sid": st["sid"], "rep": rep,
                         "x": feats, "y": y, "c": c})
    return rows


def controller_choices(train, test):
    X = np.array([r["x"] for r in train])
    w = {o: ridge_fit(X, np.array([r["y"][o] - r["y"]["stay"] for r in train]))
         for o in ("tr", "hi")}
    Xt = np.array([r["x"] for r in test])
    d_tr, d_hi = ridge_pred(w["tr"], Xt), ridge_pred(w["hi"], Xt)
    out = []
    for j in range(len(test)):
        vals = {"stay": 0.0, "tr": d_tr[j], "hi": d_hi[j]}
        out.append(max(vals, key=vals.get))
    return out, w


def crossfit(rows, by_sid=True, k=5):
    sids = sorted({r["sid"] for r in rows})
    rng = np.random.default_rng(SEED)
    order = np.array(sids)
    rng.shuffle(order)
    fold_of = {s: i % k for i, s in enumerate(order)}
    choice = [None] * len(rows)
    for f in range(k):
        tr = [r for r in rows if fold_of[r["sid"]] != f]
        te = [r for r in rows if fold_of[r["sid"]] == f]
        ch, _ = controller_choices(tr, te)
        for r, c in zip(te, ch):
            choice[r["i"]] = c
    return choice


def value(rows, choice):
    acc = np.array([r["y"][c] for r, c in zip(rows, choice)])
    cost = float(np.mean([r["c"][c] for r, c in zip(rows, choice)]))
    look = float(np.mean([c != "stay" for c in choice]))
    return acc, cost, look


def mixture_at(rows, B, fit_rows=None):
    """state-independent {stay,tr,hi} mixture with E[cost] <= B; weights from
    fit_rows means (default: rows themselves = in-sample)."""
    src = fit_rows or rows
    ybar = {o: float(np.mean([r["y"][o] for r in src])) for o in OPTS}
    cbar = {o: float(np.mean([r["c"][o] for r in src])) for o in OPTS}
    best = None
    for o in OPTS:
        if cbar[o] <= B + 1e-9 and (best is None or ybar[o] > best[1]):
            best = ({o: 1.0}, ybar[o])
    for i, a in enumerate(OPTS):
        for b in OPTS[i + 1:]:
            lo, hi = sorted((a, b), key=lambda x: cbar[x])
            if not (cbar[lo] <= B + 1e-9 < cbar[hi]):
                continue
            wt = (B - cbar[lo]) / (cbar[hi] - cbar[lo])
            v = (1 - wt) * ybar[lo] + wt * ybar[hi]
            if v > best[1]:
                best = ({lo: 1 - wt, hi: wt}, v)
    w = best[0]
    acc = np.array([sum(wt * r["y"][o] for o, wt in w.items()) for r in rows])
    return acc, w


def paired_ci(a, b):
    return sa.boot_ci(np.asarray(a) - np.asarray(b))


def main():
    states = sa.load()
    out = {}
    md = [f"# (B) History-conditioned two-step controller + (C) transfer "
          f"(dev role, n={len(states)} states x 2 replays)", "",
          "Controller features (fixed a priori): [a1 answer != STOP answer,",
          "len(a1 answer)]; two ridge Δ-heads, argmax(0, Δ̂_TR, Δ̂_HI).",
          "Same-replay decision+outcome; controller rows are 5-fold",
          "cross-fitted by state. Oracle = cross-replay (audit Q1).", ""]
    for scorer in ("strict", "lenient"):
        md += [f"## {scorer}", ""]
        res = {}
        for a1 in sa.A1S:
            rows = branch_data(states, a1, scorer)
            ch_cf = crossfit(rows)
            ctrl, ctrl_c, ctrl_look = value(rows, ch_cf)
            stay = np.array([r["y"]["stay"] for r in rows])
            tr = np.array([r["y"]["tr"] for r in rows])
            hi = np.array([r["y"]["hi"] for r in rows])
            p = ctrl_look
            rand = (1 - p) * stay + p * 0.5 * (tr + hi)
            mix_in, w_in = mixture_at(rows, ctrl_c)
            # cross-fitted mixture: weights from other fold's rows
            sids = sorted({r["sid"] for r in rows})
            rng = np.random.default_rng(SEED)
            order = np.array(sids)
            rng.shuffle(order)
            fold_of = {s: i % 2 for i, s in enumerate(order)}
            mix_cf = np.zeros(len(rows))
            for f in (0, 1):
                te = [r for r in rows if fold_of[r["sid"]] == f]
                fit = [r for r in rows if fold_of[r["sid"]] != f]
                acc, _ = mixture_at(te, ctrl_c, fit_rows=fit)
                for r, v in zip(te, acc):
                    mix_cf[r["i"]] = v
            # cross-replay sequential oracle (ceiling)
            orc = []
            for st in states:
                o = {}
                for rep in (1, 2):
                    o[rep] = {"stay": sa.sc(st["ref"][(a1, rep)], st, scorer),
                              "tr": sa.sc(st["seq"][(sa.TRAJ_OF[(a1, "crop_tr")], rep)], st, scorer),
                              "hi": sa.sc(st["seq"][(sa.TRAJ_OF[(a1, "highres")], rep)], st, scorer)}
                vals = []
                for sel, ev in ((1, 2), (2, 1)):
                    k = max(sorted(o[sel]), key=lambda k: (o[sel][k], k))
                    vals.append(o[ev][k])
                orc.append(np.mean(vals))
            ci_mix = paired_ci(ctrl, mix_cf)
            ci_stay = paired_ci(ctrl, stay)
            res[a1] = {
                "stop_after_a1": float(stay.mean()),
                "always_tr": float(tr.mean()), "always_hi": float(hi.mean()),
                "random_at_rate": float(rand.mean()),
                "mixture_insample": float(mix_in.mean()),
                "mixture_insample_w": w_in,
                "mixture_crossfit": float(mix_cf.mean()),
                "controller": float(ctrl.mean()),
                "controller_cost": ctrl_c, "controller_look": ctrl_look,
                "ctrl_minus_mixcf": float(ctrl.mean() - mix_cf.mean()),
                "ci95_vs_mixcf": ci_mix,
                "ctrl_minus_stay": float(ctrl.mean() - stay.mean()),
                "ci95_vs_stay": ci_stay,
                "oracle_crossreplay": float(np.mean(orc))}
            r = res[a1]
            md += [f"### a1 = {a1}  (controller look-rate {ctrl_look:.2f}, "
                   f"incremental cost {ctrl_c:.0f} tok)", "",
                   "| policy | acc |", "|---|--:|",
                   f"| stop after a1 | {r['stop_after_a1']:.3f} |",
                   f"| always →TR | {r['always_tr']:.3f} |",
                   f"| always →HIGHRES | {r['always_hi']:.3f} |",
                   f"| random @ rate | {r['random_at_rate']:.3f} |",
                   f"| mixture (in-sample) | {r['mixture_insample']:.3f} |",
                   f"| mixture (cross-fit) | {r['mixture_crossfit']:.3f} |",
                   f"| **controller (cross-fit)** | **{r['controller']:.3f}** |",
                   f"| sequential oracle (cross-replay) | {r['oracle_crossreplay']:.3f} |",
                   "",
                   f"controller − mixture-CF = {r['ctrl_minus_mixcf']:+.4f} "
                   f"[{ci_mix[0]:+.4f}, {ci_mix[1]:+.4f}]; "
                   f"controller − stop-after-a1 = {r['ctrl_minus_stay']:+.4f} "
                   f"[{ci_stay[0]:+.4f}, {ci_stay[1]:+.4f}]", ""]
        # (C) cross-action transfer
        md += ["### (C) cross-action transfer (heads fitted on one a1 branch, "
               "applied unchanged to the other)", "",
               "| direction | native CF acc | transferred acc | Δ (transfer − native) |",
               "|---|--:|--:|--:|"]
        tr_res = {}
        datas = {a1: branch_data(states, a1, scorer) for a1 in sa.A1S}
        cf_acc = {a1: float(value(datas[a1], crossfit(datas[a1]))[0].mean())
                  for a1 in sa.A1S}
        for src, dst in (("center", "ocr"), ("ocr", "center")):
            ch, _ = controller_choices(datas[src], datas[dst])
            acc, _, _ = value(datas[dst], ch)
            tr_res[f"{src}->{dst}"] = {"native_cf": cf_acc[dst],
                                       "transferred": float(acc.mean())}
            md.append(f"| {src}→{dst} | {cf_acc[dst]:.3f} | {acc.mean():.3f} "
                      f"| {acc.mean() - cf_acc[dst]:+.4f} |")
        md.append("")
        out[scorer] = {"branches": res, "transfer": tr_res}
    (ROOT / "paper/seq_controller.json").write_text(json.dumps(out, indent=1))
    (ROOT / "paper/seq_controller.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
