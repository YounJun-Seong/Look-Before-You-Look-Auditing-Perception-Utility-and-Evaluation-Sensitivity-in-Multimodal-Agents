#!/usr/bin/env python
"""Feature-group BASELINES, trained on development, evaluated once on the lockbox.

Reviewer question: "the gate is called state-aware, but its inputs are answer /
question surface plus confidence interactions — how much of the signal is just
cheap behavioural traces?"  This script answers it with trained baselines:

    confidence only        raw confidence rank, no training (M0)
    question surface only  logistic on q_* features
    answer surface only    logistic on ans_* / is_* / has_hedge
    Q + A, no confidence   logistic on all 11 surface features
    full recipe (retrain)  sanity: same 16 features, same recipe -> ~frozen v2.2
    full v2.2 (frozen)     the pre-registered model, unchanged

Every trained variant uses the v2.2 recipe verbatim (same development states,
same hyper-parameters, same frozen confidence CDF) so the only thing that
changes is the feature set.  NOTHING here modifies v2.2 or its verdict; these
are explanatory comparisons on the same single pass.

Outputs paper/feature_baselines_lockbox.{json,md}.
"""
from __future__ import annotations
import json, pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppu.gate_features import FEATURE_NAMES, extract  # noqa: E402

THR = 0.5
FR = pickle.load(open(ROOT / "models/gate_frozen_v2.2_pointwise.pkl", "rb"))
DEV_CDF = FR["conf_transform"]["sorted_dev_conf"]
HYPER = FR["hyper"]
DEV_SETS = [("logs/a1_vqav2_250.jsonl", "vqav2"), ("logs/a1_vqav2_confirm.jsonl", "vqav2"),
            ("logs/a1_textvqa250.jsonl", "textvqa")]
LOCK_SETS = [("logs/lockbox_textvqa.jsonl", "textvqa"), ("logs/lockbox_vqav2.jsonl", "vqav2")]

Q_COLS = [i for i, f in enumerate(FEATURE_NAMES) if f.startswith("q_")]
A_COLS = [i for i, f in enumerate(FEATURE_NAMES)
          if f.startswith("ans_") or f in ("is_numeric", "is_yesno", "has_hedge")]
C_COLS = [i for i, f in enumerate(FEATURE_NAMES) if f.startswith("conf_x_")]
ALL = list(range(len(FEATURE_NAMES)))


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0: return "HELP"
    if d1 < 0 and d2 < 0: return "HARM"
    if d1 == 0 and d2 == 0: return "NEUTRAL"
    return "AMBIGUOUS"


def conf_pct(x):
    return float(1.0 - np.searchsorted(DEV_CDF, x) / len(DEV_CDF))


def load(log, source):
    by = defaultdict(dict)
    for line in open(ROOT / log):
        r = json.loads(line)
        if r.get("type") == "rollout":
            by[r["sample_id"]][(r["action"]["type"], r["replay"])] = r
    out = []
    for sid, d in by.items():
        if not all(k in d for k in [("STOP", 1), ("STOP", 2), ("HIGHRES", 1), ("HIGHRES", 2)]):
            continue
        s1, s2 = d[("STOP", 1)]["score_after"], d[("STOP", 2)]["score_after"]
        x1, x2 = d[("HIGHRES", 1)]["score_after"], d[("HIGHRES", 2)]["score_after"]
        st = d[("STOP", 1)]
        out.append({"state_id": sid, "source": source, "label": four_way(s1, s2, x1, x2),
                    "answer": st["answer_after"], "question": st["question"],
                    "answer_len_tokens": st["cost"]["tokens_out"],
                    "stop": (s1 + s2) / 2, "hi": (x1 + x2) / 2,
                    "hi_tokens": d[("HIGHRES", 1)]["cost"]["visual_tokens"],
                    "mean_logprob": float(np.mean(
                        [d[("STOP", r)]["confidence"]["mean_logprob"] for r in (1, 2)]))})
    return out


def featurize(rows):
    conf = np.array([conf_pct(r["mean_logprob"]) for r in rows])
    X = np.array([extract(r, c) for r, c in zip(rows, conf)], float)
    y = np.array([r["label"] == "HELP" for r in rows], int)
    return X, y, conf


dev = [r for log, s in DEV_SETS for r in load(log, s) if r["label"] != "AMBIGUOUS"]
Xd, yd, cd = featurize(dev)
sd = np.array([r["source"] for r in dev])
assert len(dev) == FR["train"]["n"], (len(dev), FR["train"]["n"])


def fit(cols):
    sc = StandardScaler().fit(Xd[:, cols])
    m = LogisticRegression(**HYPER).fit(sc.transform(Xd[:, cols]), yd)
    return lambda X: m.predict_proba(sc.transform(X[:, cols]))[:, 1]


def dev_oof(cols):
    """5-fold OOF ROC on development, per source — reference only."""
    oof = np.zeros(len(yd))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=1).split(Xd, yd):
        sc = StandardScaler().fit(Xd[tr][:, cols])
        m = LogisticRegression(**HYPER).fit(sc.transform(Xd[tr][:, cols]), yd[tr])
        oof[te] = m.predict_proba(sc.transform(Xd[te][:, cols]))[:, 1]
    return {s: roc_auc_score(yd[sd == s], oof[sd == s]) for s in ("textvqa", "vqav2")}


VARIANTS = [
    ("confidence only (no training)", None),
    ("question surface only", Q_COLS),
    ("answer surface only", A_COLS),
    ("question + answer, no confidence", Q_COLS + A_COLS),
    ("full recipe, retrained (sanity)", ALL),
    ("full v2.2 (frozen)", "frozen"),
]

rng = np.random.default_rng(20260822)
results = {}
for log, source in LOCK_SETS:
    rows = [r for r in load(log, source) if r["label"] != "AMBIGUOUS"]
    X, y, conf = featurize(rows)
    prev = y.mean()
    stop = np.array([r["stop"] for r in rows]); hi = np.array([r["hi"] for r in rows])
    tok = np.array([r["hi_tokens"] for r in rows], float)

    def lift(s):
        k = max(1, int(round(prev * len(y))))
        return (y[np.argsort(-s)[:k]].sum() / max(y.sum(), 1)) / (k / len(y))

    def ci(s, B=4000):
        vals = []
        for _ in range(B):
            i = rng.integers(0, len(y), len(y))
            if y[i].sum() == 0: continue
            k = max(1, int(round(prev * len(i))))
            vals.append((y[i][np.argsort(-s[i])[:k]].sum() / y[i].sum()) / (k / len(i)))
        return [float(v) for v in np.quantile(vals, [0.025, 0.975])]

    def budget_acc(s, b):
        k = max(1, int(round(b * len(y))))
        pick = np.zeros(len(y), bool); pick[np.argsort(-s)[:k]] = True
        return float(np.where(pick, hi, stop).mean()), float(y[pick].sum() / max(y.sum(), 1))

    results[source] = {"n": len(rows), "prevalence": float(prev),
                       "always_stop": float(stop.mean()), "always_highres": float(hi.mean()),
                       "variants": {}}
    for name, cols in VARIANTS:
        if cols is None:
            s = conf
            oof = None
        elif cols == "frozen":
            s = FR["model"].predict_proba(FR["scaler"].transform(X))[:, 1]
            oof = None
        else:
            s = fit(cols)(X)
            oof = dev_oof(cols)
        lo, hi_ci = ci(s)
        results[source]["variants"][name] = {
            "roc": float(roc_auc_score(y, s)),
            "pr_over_prev": float(average_precision_score(y, s) / prev),
            "lift": float(lift(s)), "lift_ci": [lo, hi_ci],
            "acc@10%": budget_acc(s, 0.10)[0], "acc@20%": budget_acc(s, 0.20)[0],
            "recall@20%": budget_acc(s, 0.20)[1],
            "dev_oof_roc": oof,
        }

# ---------------------------------------------------------------- report
out_json = ROOT / "paper/feature_baselines_lockbox.json"
out_md = ROOT / "paper/feature_baselines_lockbox.md"
json.dump(results, open(out_json, "w"), indent=1)

L = ["# Feature-group baselines on the lockbox (single pass, trained on development)", "",
     "All trained variants: v2.2 recipe verbatim (same 601 development states, same "
     "hyper-parameters, frozen development confidence CDF); only the feature set changes. "
     "v2.2 itself is untouched. Budget accuracies are replay-averaged task accuracy when "
     "the top-b fraction of states by score is sent to HIGHRES.", ""]
for source in ("textvqa", "vqav2"):
    R = results[source]
    L += [f"## {source.upper()}  (n={R['n']}, prevalence {100*R['prevalence']:.1f}%, "
          f"always-STOP {100*R['always_stop']:.1f}% / always-HIGHRES {100*R['always_highres']:.1f}%)", "",
          "| variant | ROC | PR/prev | lift@1× | 95% CI | acc@10% | acc@20% | HELP recall@20% | dev OOF ROC (tv / vqa) |",
          "|---|--:|--:|--:|:--:|--:|--:|--:|:--:|"]
    for name, v in R["variants"].items():
        oof = ("—" if v["dev_oof_roc"] is None else
               f"{v['dev_oof_roc']['textvqa']:.3f} / {v['dev_oof_roc']['vqav2']:.3f}")
        L.append(f"| {name} | {v['roc']:.3f} | {v['pr_over_prev']:.2f}× | {v['lift']:.2f}× | "
                 f"[{v['lift_ci'][0]:.2f}, {v['lift_ci'][1]:.2f}] | {100*v['acc@10%']:.1f}% | "
                 f"{100*v['acc@20%']:.1f}% | {100*v['recall@20%']:.1f}% | {oof} |")
    L.append("")
open(out_md, "w").write("\n".join(L))
print("\n".join(L))
print(f"\nwrote {out_md}, {out_json}")
