#!/usr/bin/env python3
"""(B) Cost-aware sequential controller v2 — train on dev-100, FREEZE.

Fixes the two disclosed gaps of controller v1: no cost term, and a
2-feature history. Decision now matches the paper's framework:

    a2 = argmax_a [ dhat(a | h1) - lambda * c(a) ],  a in {stay, ->TR, ->HI}

with dhat from two ridge Δ-heads over the expanded post-a1 observables
(all deployable):

    x = [ answer_changed, len(a1 answer),
          pre_conf (STOP mean logprob), post_conf (a1 mean logprob),
          conf_change, post_min_logprob, q_has_ocr_cue ]

pre/post confidence come from the byte-match-checked regenerations
(stage-A STOP side collection + data/span_conf/seqdev_a1_conf.jsonl).
lambda is frozen on dev per branch at B = the frozen-mixture matched cost
(always->TR cost), plus a disclosed lambda=0 (cost-blind) variant.
5-fold cross-fitted dev numbers are reported; the frozen artifact refits on
all 100 states. Writes models/seq_controller_v2_frozen.{pkl,json}.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "sqc", ROOT / "scripts/seq_controller.py")
sqc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sqc)
sa = sqc.sa

SEED = 20260901
OPTS = ("stay", "tr", "hi")
LAM = [0.0] + list(np.logspace(-6, -2, 25))
OCR_CUE = re.compile(r"\b(say|says|written|word|text|sign|read|letter|number|"
                     r"brand|name|title|label)\b", re.I)


def load_conf():
    stopc = {}
    for line in open(ROOT / "data/stage_a_features.jsonl"):
        r = json.loads(line)
        stopc[r["state_id"]] = {
            rep: r[f"stop_r{rep}"] for rep in (1, 2)
            if r.get(f"stop_r{rep}") and r[f"stop_r{rep}"].get("mean_logprob")
            is not None}
    a1c = {}
    for line in open(ROOT / "data/span_conf/seqdev_a1_conf.jsonl"):
        r = json.loads(line)
        if r.get("type") == "a1_conf":
            a1c[r["sample_id"]] = r
    return stopc, a1c


def branch_rows(states, a1, scorer, stopc, a1c):
    rows = []
    for st in states:
        sid = st["sid"]
        if sid not in a1c or sid not in stopc:
            continue
        for rep in (1, 2):
            pre = stopc[sid].get(rep, {}).get("mean_logprob")
            ac = a1c[sid].get(f"{a1}_r{rep}") or {}
            post, pmin = ac.get("mean_logprob"), ac.get("min_logprob")
            if pre is None or post is None:
                continue
            a_ans = st["ref"][(a1, rep)]["answer_after"] or ""
            s_ans = st["ref"][("stop", rep)]["answer_after"] or ""
            x = [float(sa.normalize_answer(a_ans) != sa.normalize_answer(s_ans)),
                 float(len(a_ans)), float(pre), float(post),
                 float(post - pre), float(pmin),
                 1.0 if OCR_CUE.search(st["ref"][("stop", rep)]["question"] or "")
                 else 0.0]
            y = {"stay": sa.sc(st["ref"][(a1, rep)], st, scorer),
                 "tr": sa.sc(st["seq"][(sa.TRAJ_OF[(a1, "crop_tr")], rep)], st, scorer),
                 "hi": sa.sc(st["seq"][(sa.TRAJ_OF[(a1, "highres")], rep)], st, scorer)}
            a1cost = st["ref"][(a1, rep)]["cost"]["visual_tokens"]
            c = {"stay": 0.0,
                 "tr": st["seq"][(sa.TRAJ_OF[(a1, "crop_tr")], rep)]["cost"]["visual_tokens"] - a1cost,
                 "hi": st["seq"][(sa.TRAJ_OF[(a1, "highres")], rep)]["cost"]["visual_tokens"] - a1cost}
            rows.append({"i": len(rows), "sid": sid, "x": x, "y": y, "c": c})
    return rows


def fit_heads(rows):
    X = np.array([r["x"] for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w = {}
    for o in ("tr", "hi"):
        y = np.array([r["y"][o] - r["y"]["stay"] for r in rows])
        A = np.hstack([Xs, np.ones((len(Xs), 1))])
        w[o] = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ y)
    return {"mu": mu, "sd": sd, "w": w}


def choices(model, rows, lam):
    X = np.array([r["x"] for r in rows])
    Xs = (X - model["mu"]) / model["sd"]
    A = np.hstack([Xs, np.ones((len(Xs), 1))])
    d = {o: A @ model["w"][o] for o in ("tr", "hi")}
    out = []
    for j, r in enumerate(rows):
        vals = {"stay": 0.0,
                "tr": d["tr"][j] - lam * r["c"]["tr"],
                "hi": d["hi"][j] - lam * r["c"]["hi"]}
        out.append(max(vals, key=vals.get))
    return out


def value(rows, ch):
    acc = np.array([r["y"][c] for r, c in zip(rows, ch)])
    cost = float(np.mean([r["c"][c] for r, c in zip(rows, ch)]))
    look = float(np.mean([c != "stay" for c in ch]))
    return acc, cost, look


def crossfit(rows, lam, k=5):
    sids = sorted({r["sid"] for r in rows})
    rng = np.random.default_rng(SEED)
    order = np.array(sids)
    rng.shuffle(order)
    fold_of = {s: i % k for i, s in enumerate(order)}
    ch = [None] * len(rows)
    for f in range(k):
        tr = [r for r in rows if fold_of[r["sid"]] != f]
        te = [r for r in rows if fold_of[r["sid"]] == f]
        m = fit_heads(tr)
        for r, c in zip(te, choices(m, te, lam)):
            ch[r["i"]] = c
    return ch


def main():
    states = sa.load()
    stopc, a1c = load_conf()
    blob = {"features": ["answer_changed", "a1_answer_len", "pre_conf",
                         "post_conf", "conf_change", "post_min_logprob",
                         "q_has_ocr_cue"],
            "decision": "argmax_a [dhat(a|h1) - lambda*c(a)]; ridge alpha=1",
            "heads": {}, "lambda": {}, "seed": SEED, "dev_table": {}}
    for scorer in ("strict",):
        for a1 in sa.A1S:
            rows = branch_rows(states, a1, scorer, stopc, a1c)
            print(f"== {a1}: {len(rows)} rows (of 200)")
            # freeze lambda at B = always->TR cost (the frozen mixture's cost)
            B = float(np.mean([r["c"]["tr"] for r in rows]))
            best = (0.0, -1.0)
            m_all = fit_heads(rows)
            for lam in LAM:
                acc, cost, _ = value(rows, choices(m_all, rows, lam))
                if cost <= B + 1e-9 and acc.mean() > best[1]:
                    best = (float(lam), float(acc.mean()))
            lam_star = best[0]
            dev = {}
            for name, lam in (("lam0", 0.0), ("lamB", lam_star)):
                ch = crossfit(rows, lam)
                acc, cost, look = value(rows, ch)
                tr_acc = np.array([r["y"]["tr"] for r in rows])
                stay = np.array([r["y"]["stay"] for r in rows])
                d = acc - tr_acc
                ci = sa.boot_ci(d)
                dev[name] = {"lam": lam, "acc": float(acc.mean()),
                             "cost": cost, "look": look,
                             "vs_mixture_tr": float(d.mean()), "ci95": ci,
                             "vs_stay": float((acc - stay).mean())}
                print(f"  {name} (λ={lam:.2e}): acc {acc.mean():.3f} cost "
                      f"{cost:.0f} look {look:.2f} | vs always→TR "
                      f"{d.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]")
            blob["heads"][a1] = {k: v.tolist() if hasattr(v, "tolist") else
                                 {o: w.tolist() for o, w in v.items()}
                                 for k, v in fit_heads(rows).items()}
            blob["lambda"][a1] = {"lam0": 0.0, "lamB": lam_star, "B": B}
            blob["dev_table"][a1] = dev
    with open(ROOT / "models/seq_controller_v2_frozen.pkl", "wb") as fh:
        pickle.dump(blob, fh)
    sha = hashlib.sha256((ROOT / "models/seq_controller_v2_frozen.pkl")
                         .read_bytes()).hexdigest()[:16]
    (ROOT / "models/seq_controller_v2_frozen.json").write_text(
        json.dumps({"artifact": "models/seq_controller_v2_frozen.pkl",
                    "sha256_16": sha, **{k: v for k, v in blob.items()
                                         if k != "heads"}}, indent=1))
    print("frozen -> models/seq_controller_v2_frozen.pkl", sha)


if __name__ == "__main__":
    main()
