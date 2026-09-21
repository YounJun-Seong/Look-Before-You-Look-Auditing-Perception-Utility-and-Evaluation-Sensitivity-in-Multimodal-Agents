#!/usr/bin/env python
"""Gate v2: matched pairwise ranking, built to beat confidence where confidence fails.

v1 optimised separability (ROC 0.671) but not the operating point we can afford
(lift 1.2-1.4x at a 15% LOOK budget). That is an objective mismatch, so v2
learns the ordering directly: for a HELP state p and a non-HELP state n, the
score of p must exceed the score of n.

Pairs are MATCHED — same answer type, nearby answer-length and confidence bins,
same source where possible. An unmatched pair (long TextVQA HELP vs short
yes/no negative) is solvable by surface form alone and teaches nothing.

Half the pairs are hard by construction:
    hard positive   HELP with HIGH confidence      (confidence would stop)
    hard negative   non-HELP with LOW confidence   (confidence would look)
These two sets are the entire reason v2 should beat a confidence threshold.
"""
from __future__ import annotations
import json, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppu.gate_features import FEATURE_NAMES, extract

THR = 0.5
RNG = np.random.default_rng(20260818)


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0: return "HELP"
    if d1 < 0 and d2 < 0: return "HARM"
    if d1 == 0 and d2 == 0: return "NEUTRAL"
    return "AMBIGUOUS"


def load(log, manifest, source):
    man = {json.loads(l)["sample_id"]: json.loads(l) for l in open(manifest)}
    by = defaultdict(dict)
    for line in open(log):
        r = json.loads(line)
        if r.get("type") == "rollout":
            by[r["sample_id"]][(r["action"]["type"], r["replay"])] = r
    out = []
    for sid, d in by.items():
        if not all(k in d for k in [("STOP", 1), ("STOP", 2), ("HIGHRES", 1), ("HIGHRES", 2)]):
            continue
        s1, s2 = d[("STOP", 1)]["score_after"], d[("STOP", 2)]["score_after"]
        x1, x2 = d[("HIGHRES", 1)]["score_after"], d[("HIGHRES", 2)]["score_after"]
        st = d[("STOP", 1)]; meta = (man.get(sid) or {}).get("meta") or {}
        out.append({"state_id": sid, "source": source, "label": four_way(s1, s2, x1, x2),
                    "stratum": meta.get("stratum", "-"), "qtype": meta.get("qtype", "-"),
                    "answer": st["answer_after"], "question": st["question"],
                    "answer_len_tokens": st["cost"]["tokens_out"],
                    "mean_logprob": float(np.mean([d[("STOP", r)]["confidence"]["mean_logprob"] for r in (1, 2)])),
                    "min_logprob": float(np.mean([d[("STOP", r)]["confidence"]["min_logprob"] for r in (1, 2)]))})
    return out


SETS = [("logs/a1_vqav2_250.jsonl", "data/manifests/vqav2_a1.jsonl", "vqav2"),
        ("logs/a1_vqav2_confirm.jsonl", "data/manifests/vqav2_confirm.jsonl", "vqav2"),
        ("logs/a1_textvqa250.jsonl", "data/manifests/textvqa_train.jsonl", "textvqa")]
rows = [r for log, man, src in SETS for r in load(log, man, src)]
rows = [r for r in rows if r["label"] != "AMBIGUOUS"]

lp = np.array([r["mean_logprob"] for r in rows])
order = lp.argsort().argsort() / max(len(lp) - 1, 1)
for r, cr in zip(rows, 1 - order):          # 1 = least confident
    r["conf_rank"] = float(cr)
    r["y"] = int(r["label"] == "HELP")
X_all = np.array([extract(r, r["conf_rank"]) for r in rows], float)
y_all = np.array([r["y"] for r in rows])
print(f"states {len(rows)}  HELP {int(y_all.sum())}  "
      f"(vqav2 {sum(1 for r in rows if r['source']=='vqav2')}, "
      f"textvqa {sum(1 for r in rows if r['source']=='textvqa')})")


def bins(rows, idx):
    """Matching keys: answer type, length bin, confidence bin."""
    out = {}
    for i in idx:
        r = rows[i]
        a = r["answer"]
        t = ("num" if a.strip().replace(".", "").isdigit()
             else "yesno" if a.strip().lower()[:3] in ("yes", "no") else "other")
        out[i] = (t, min(len(a) // 5, 4), min(int(r["conf_rank"] * 4), 3))
    return out


def make_pairs(rows, idx, n_pairs=4000, hard_frac=0.25):
    pos = [i for i in idx if rows[i]["y"] == 1]
    neg = [i for i in idx if rows[i]["y"] == 0]
    if not pos or not neg:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros(0), np.zeros(0)
    key = bins(rows, idx)
    conf = {i: rows[i]["conf_rank"] for i in idx}
    hard_pos = [i for i in pos if conf[i] < 0.5]      # confident yet HELP
    hard_neg = [i for i in neg if conf[i] >= 0.5]     # unconfident yet no gain
    # Negatives indexed by (source, matching key): a pair drawn across sources
    # is solvable by surface form alone.
    by_src_key = defaultdict(list)
    for i in neg:
        by_src_key[(rows[i]["source"], key[i])].append(i)
    by_src = defaultdict(list)
    for i in neg:
        by_src[rows[i]["source"]].append(i)

    pairs = []
    n_hard = int(n_pairs * hard_frac)
    for k in range(n_pairs):
        hard = k < n_hard and hard_pos and hard_neg
        p = int(RNG.choice(hard_pos if hard else pos))
        src = rows[p]["source"]
        cands = by_src_key.get((src, key[p]), []) or by_src.get(src, []) or neg
        if hard:
            hn = set(hard_neg)
            m = [i for i in cands if i in hn]
            cands = m or [i for i in by_src.get(src, []) if i in hn] or cands
        n = int(RNG.choice(cands))
        pairs.append((p, n))

    Xp = np.array([extract(rows[p], rows[p]["conf_rank"]) for p, _ in pairs], float)
    Xn = np.array([extract(rows[n], rows[n]["conf_rank"]) for _, n in pairs], float)
    D = np.vstack([Xp - Xn, Xn - Xp])
    t = np.r_[np.ones(len(pairs)), np.zeros(len(pairs))]
    # Source-balanced loss: each source contributes half, whatever its count.
    src_of = [rows[p]["source"] for p, _ in pairs] * 2
    cnt = Counter(src_of)
    w = np.array([1.0 / cnt[s] for s in src_of]) * (len(src_of) / max(len(cnt), 1))
    return D, t, w


def fit_rank(rows, idx, scaler):
    D, t, w = make_pairs(rows, idx)
    if len(D) == 0:
        return None
    Ds = D / (scaler.scale_ + 1e-9)                  # differences: scale only
    m = LogisticRegression(max_iter=5000, C=0.3, fit_intercept=False)
    m.fit(Ds, t, sample_weight=w)
    return m


def lift(y, s, prev, mult=1.0):
    k = max(1, int(round(prev * mult * len(y))))
    return (y[np.argsort(-s)[:k]].sum() / max(y.sum(), 1)) / (k / len(y))


oof = np.zeros(len(rows))
skf = StratifiedKFold(5, shuffle=True, random_state=1)
for tr, te in skf.split(X_all, y_all):
    sc = StandardScaler().fit(X_all[tr])
    m = fit_rank(rows, list(tr), sc)
    oof[te] = (X_all[te] / (sc.scale_ + 1e-9)) @ m.coef_[0]

conf_score = np.array([r["conf_rank"] for r in rows])
v1 = pickle.load(open("models/gate_frozen_v1.pkl", "rb"))
X_v1 = np.array([[r[f] if f in r else 0.0 for f in v1["features"]] for r in rows], float)
for i, r in enumerate(rows):                        # v1 needs its own feature set
    X_v1[i] = [r["mean_logprob"], r["min_logprob"], r["answer_len_tokens"],
               len(r["answer"]), len(r["question"]), 0.0]
v1_score = v1["model"].predict_proba(v1["scaler"].transform(X_v1))[:, 1]

print(f"\n{'set':24s} {'n':>4s} {'HELP':>5s} {'scorer':12s} {'ROC':>6s} "
      f"{'PR/prev':>8s} {'lift@1x':>8s} {'@1.5x':>7s}")
def report(name, mask):
    y = y_all[mask]
    if y.sum() < 5: return
    prev = y.mean()
    for nm, s in (("confidence", conf_score[mask]), ("v1", v1_score[mask]),
                  ("v2 (ranking)", oof[mask])):
        print(f"{name if nm=='confidence' else '':24s} {mask.sum():4d} {int(y.sum()):5d} "
              f"{nm:12s} {roc_auc_score(y, s):6.3f} "
              f"{average_precision_score(y, s)/prev:7.1f}x {lift(y, s, prev):7.1f}x "
              f"{lift(y, s, prev, 1.5):6.1f}x")

src = np.array([r["source"] for r in rows])
strat = np.array([r["stratum"] for r in rows])
report("VQAv2 (all 500)", src == "vqav2")
for st in ["A_global_easy", "B_local_detail", "C_multi_relational"]:
    report(f"  {st}", (src == "vqav2") & (strat == st))
report("TextVQA", src == "textvqa")

sc = StandardScaler().fit(X_all)
final = fit_rank(rows, list(range(len(rows))), sc)
pickle.dump({"version": "gate_v2.1_rank", "features": FEATURE_NAMES,
             "scale": sc.scale_, "coef": final.coef_[0],
             "protocol": {"score": "x/scale @ coef; higher = LOOK",
                          "trained_on": "VQAv2 500 + TextVQA 250 (dev/burn)",
                          "pairs": "matched within source, then answer type/length/"
                                   "confidence bin; 25% hard; source-balanced loss",
                          "refit_on_lockbox": False}},
            open("models/gate_v2_1_rank.pkl", "wb"))
print("\n-> models/gate_v2_1_rank.pkl")
print("top coefficients:", ", ".join(
    f"{n}{c:+.2f}" for n, c in sorted(zip(FEATURE_NAMES, final.coef_[0]),
                                      key=lambda t: -abs(t[1]))[:6]))
