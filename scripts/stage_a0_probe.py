#!/usr/bin/env python
"""Stage A0 learnability probe: is there any signal a gate could use?

396 states is not enough to claim performance, so nothing here is a
significance test. The question is whether spending GPU on A1 is justified,
and the number that answers it is the one printed first:

    at a LOOK rate matched to the positive prevalence, how many more useful
    states does a scorer catch than a random gate?

Positives are the replicate-consistent LOOK states; UNCERTAIN is excluded
(reported separately, never pushed into 0/1).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

LAB = "data/stage_a_labels_highres.jsonl"
FEA = "data/stage_a_features.jsonl"


def load_rows():
    lab = {json.loads(l)["state_id"]: json.loads(l) for l in open(LAB)}
    rows = []
    for line in open(FEA):
        f = json.loads(line)
        l = lab[f["state_id"]]
        rows.append({**l, "f": f})
    return rows


def pct_rank(x, group):
    """Percentile rank within a group — logprob calibration differs between a
    single-letter MCQ answer and a free-form sequence, so raw values from the
    two datasets must not share a threshold."""
    out = np.zeros(len(x))
    for g in set(group):
        m = np.array([gg == g for gg in group])
        v = np.array(x)[m]
        order = v.argsort().argsort()
        out[m] = order / max(len(v) - 1, 1)
    return out


def recall_at_rate(scores, y, rate):
    k = max(1, int(round(rate * len(y))))
    top = np.argsort(-np.asarray(scores))[:k]
    hit = int(np.asarray(y)[top].sum())
    return hit / max(y.sum(), 1), hit / k, k


def main():
    rows = load_rows()
    used = [r for r in rows if r["label"] in ("LOOK", "STOP")]
    y = np.array([1 if r["label"] == "LOOK" else 0 for r in used])
    ds = [r["dataset"] for r in used]
    prev = y.mean()

    mean_lp = np.array([np.mean([r["f"][f"stop_r{i}"]["mean_logprob"] or -9
                                 for i in (1, 2)]) for r in used])
    min_lp = np.array([np.mean([r["f"][f"stop_r{i}"]["min_logprob"] or -9
                                for i in (1, 2)]) for r in used])
    need = np.array([1.0 if r["f"]["self_assessment"]["needs_detail"] else 0.0
                     for r in used])
    sa_conf = np.array([r["f"]["self_assessment"]["mean_logprob"] or -9 for r in used])
    ans_len = np.array([r["answer_len_tokens"] for r in used], dtype=float)
    q_len = np.array([r["question_len_chars"] for r in used], dtype=float)
    ratio = np.array([r["budget_ratio"] for r in used], dtype=float)

    # self-assessment as a ranked score: NEED_DETAIL first, ties broken by how
    # confidently it said so, so it can be evaluated at a matched LOOK rate.
    sa_score = need + (need * 2 - 1) * (sa_conf - sa_conf.min()) / (
        np.ptp(sa_conf) + 1e-9) * 0.49

    scorers = {
        "random (expected)": None,
        "confidence (raw -mean logprob)": -mean_lp,
        "confidence (dataset-wise rank)": 1 - pct_rank(mean_lp, ds),
        "min-token logprob (rank)": 1 - pct_rank(min_lp, ds),
        "self-assessment (ranked)": sa_score,
        "answer length": ans_len,
    }

    # M1: logistic regression, out-of-fold. Two variants — with and without the
    # dataset-wise normalization, since that feature uses dataset identity.
    feats = {
        "M1 logistic (no dataset info)": np.c_[mean_lp, min_lp, need, sa_conf,
                                               ans_len, q_len, ratio],
        "M1 logistic (+dataset-wise rank)": np.c_[
            mean_lp, min_lp, need, sa_conf, ans_len, q_len, ratio,
            pct_rank(mean_lp, ds), pct_rank(min_lp, ds)],
    }
    for name, X in feats.items():
        oof = np.zeros(len(y))
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=2000, class_weight="balanced",
                                   C=0.5).fit(sc.transform(X[tr]), y[tr])
            oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
        scorers[name] = oof

    print(f"states used {len(used)} (LOOK {int(y.sum())}, STOP {int((1-y).sum())}), "
          f"UNCERTAIN excluded {sum(1 for r in rows if r['label']=='UNCERTAIN')}")
    print(f"positive prevalence {100*prev:.1f}%  -> matched LOOK rate\n")
    print(f"{'scorer':34s} {'recall':>7s} {'lift':>6s} {'prec':>6s} {'PR-AUC':>7s} {'/prev':>6s}")
    for name, s in scorers.items():
        if s is None:
            print(f"{'random (expected)':34s} {100*prev:6.1f}% {1.0:5.1f}x "
                  f"{100*prev:5.1f}% {prev:7.3f} {1.0:5.1f}x")
            continue
        rec, prec, k = recall_at_rate(s, y, prev)
        ap = average_precision_score(y, s)
        print(f"{name:34s} {100*rec:6.1f}% {rec/prev:5.1f}x {100*prec:5.1f}% "
              f"{ap:7.3f} {ap/prev:5.1f}x")

    print("\nby dataset (matched rate = that dataset's prevalence)")
    for d in ("mme", "textvqa"):
        m = np.array([x == d for x in ds])
        yd, p = y[m], y[m].mean()
        print(f"  {d}: n={m.sum()} prevalence {100*p:.1f}%")
        for name in ("confidence (dataset-wise rank)", "self-assessment (ranked)",
                     "M1 logistic (no dataset info)"):
            s = np.asarray(scorers[name])[m]
            rec, prec, k = recall_at_rate(s, yd, p)
            print(f"    {name:32s} recall {100*rec:5.1f}%  lift {rec/max(p,1e-9):4.1f}x"
                  f"  PR-AUC {average_precision_score(yd, s):.3f}")

    print("\nscore distribution by outcome group (confidence rank, higher = less confident)")
    conf_rank = 1 - pct_rank(mean_lp, ds)
    groups = {"HELP (LOOK)": y == 1,
              "HARM": np.array([r["harmful"] for r in used]),
              "NEUTRAL (rest)": (y == 0) & ~np.array([r["harmful"] for r in used])}
    for g, m in groups.items():
        if m.sum():
            print(f"  {g:16s} n={int(m.sum()):4d}  mean {conf_rank[m].mean():.3f}  "
                  f"median {np.median(conf_rank[m]):.3f}")
    sa = np.array([r["f"]["self_assessment"]["needs_detail"] for r in used])
    for g, m in groups.items():
        if m.sum():
            print(f"  {g:16s} self-assessment NEED_DETAIL rate {100*sa[m].mean():.1f}%")


if __name__ == "__main__":
    sys.exit(main())
