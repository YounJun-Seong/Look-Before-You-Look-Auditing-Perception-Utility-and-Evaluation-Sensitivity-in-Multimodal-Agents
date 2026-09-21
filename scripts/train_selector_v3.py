#!/usr/bin/env python3
"""Selector v3 — hidden-state representation controller; train on dev, FREEZE.

Same decision form and dev protocol as v2 (per-family ΔScore heads,
argmax_f dhat_f - lambda(B) * cost_f, lambda frozen on the TextVQA dev
subset), but the candidate grid now includes the reader's own pre-action
hidden state H (mean-pooled last layer of the STOP-time input; extracted by
scripts/extract_hidden_states.py — deployable information only):

  features: {H, H+F0}   heads: {ridge_3, ridge_30, mlp64}   (+ v2's F0/ridge0.3
  reference row) — the FULL grid is disclosed; ONE winner selected by the
  same 5-fold dev CV of realized TextVQA policy value at the 1/2x budget.

Writes models/selector_frozen_v3.pkl (+ .json manifest with the grid table
and frozen lambda(B)). Run before any v3-prospective state is collected.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "tsv", ROOT / "scripts/train_selector_v2.py")
tsv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tsv)
mas = tsv.mas

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SEED = 20260901
FAMILIES = ("CROP", "OCR", "HIGHRES")
LAM_TOK = [0.0] + list(np.logspace(-6, -2, 33))
BUDGET_FRACS = (0.25, 0.5, 1.0)


def load_H():
    ids, H = [], []
    for f in ("data/hidden/dev_mme.npz", "data/hidden/dev_textvqa.npz"):
        z = np.load(ROOT / f, allow_pickle=True)
        ids += list(z["ids"])
        H.append(z["H"].astype(np.float32))
    return dict(zip(ids, np.vstack(H)))


def head_factory(name):
    if name.startswith("ridge"):
        return Ridge(alpha=float(name.split("_")[1]), random_state=0)
    if name == "mlp64":
        return MLPRegressor(hidden_layer_sizes=(64,), max_iter=800,
                            random_state=0, early_stopping=False)
    raise ValueError(name)


def build_X(states, feat_set, Hmap, ranks):
    F0 = mas.feats(states, ranks["mean"])
    H = np.stack([Hmap[s["sid"]] for s in states])
    if feat_set == "H":
        return H
    if feat_set == "H+F0":
        return np.hstack([H, F0])
    return F0


def cv_value(states, feat_set, head_name, Hmap, ranks, assign):
    tv = [s for s in states if s["dataset"] == "textvqa"]
    hi_tok = float(np.mean([s["cost_vtok"]["HIGHRES"] for s in tv]))
    X = build_X(states, feat_set, Hmap, ranks)
    vals = {f: [] for f in BUDGET_FRACS}
    for fold in range(mas.N_FOLDS):
        tr = [i for i, a in enumerate(assign) if a != fold]
        te = [i for i, a in enumerate(assign) if a == fold]
        scaler = StandardScaler().fit(X[tr])
        dh_tr, dh_te = {}, {}
        for f, key in (("CROP", "crop"), ("OCR", "ocr"), ("HIGHRES", "hi")):
            y = np.array([np.mean(states[i][key] - states[i]["stop"]) for i in tr])
            h = head_factory(head_name).fit(scaler.transform(X[tr]), y)
            dh_tr[f] = h.predict(scaler.transform(X[tr]))
            dh_te[f] = h.predict(scaler.transform(X[te]))
        tr_tv = [j for j, i in enumerate(tr) if states[i]["dataset"] == "textvqa"]
        te_tv = [j for j, i in enumerate(te) if states[i]["dataset"] == "textvqa"]
        s_tr = [states[tr[j]] for j in tr_tv]
        s_te = [states[te[j]] for j in te_tv]
        d_tr = {f: dh_tr[f][tr_tv] for f in FAMILIES}
        d_te = {f: dh_te[f][te_tv] for f in FAMILIES}
        for frac in BUDGET_FRACS:
            B = frac * hi_tok
            best_lam, best_v = 0.0, -1.0
            for lam in LAM_TOK:
                ev = tsv.policy_value(s_tr, d_tr, lam)
                if ev["vtok"] <= B + 1e-9 and ev["acc"] > best_v:
                    best_lam, best_v = lam, ev["acc"]
            ev = tsv.policy_value(s_te, d_te, best_lam)
            vals[frac].append((ev["acc"], ev["vtok"], len(s_te)))
    out = {}
    for frac, rows in vals.items():
        n = sum(r[2] for r in rows)
        out[frac] = {"acc": sum(r[0] * r[2] for r in rows) / n,
                     "vtok": sum(r[1] * r[2] for r in rows) / n}
    return out


def main():
    states, dropped = mas.load_states()
    assert dropped == 0 and len(states) == 396
    Hmap = load_H()
    missing = [s["sid"] for s in states if s["sid"] not in Hmap]
    assert not missing, f"missing embeddings: {len(missing)}"
    ranks = {"mean": tsv.make_rank([s["mean_logprob"] for s in states])}
    assign = mas.folds(states)

    grid = [(fs, hn) for fs in ("H", "H+F0")
            for hn in ("ridge_3", "ridge_30", "mlp64")]
    table = []
    for fs, hn in grid:
        v = cv_value(states, fs, hn, Hmap, ranks, assign)
        table.append({"features": fs, "head": hn,
                      **{f"cv_acc_{f}x": round(v[f]["acc"], 4) for f in BUDGET_FRACS},
                      **{f"cv_vtok_{f}x": round(v[f]["vtok"], 1) for f in BUDGET_FRACS}})
        print(table[-1])
    v2ref = next(r for r in json.loads(
        (ROOT / "models/selector_frozen_v2.json").read_text())["candidate_table"]
        if r["features"] == "F0" and r["head"] == "ridge_0.3")
    print("v2 reference:", v2ref)
    key, tie = "cv_acc_0.5x", "cv_acc_0.25x"
    winner = max(table, key=lambda r: (r[key], r[tie]))
    print("\nWINNER:", winner)

    X = build_X(states, winner["features"], Hmap, ranks)
    scaler = StandardScaler().fit(X)
    heads = {}
    for f, k in (("CROP", "crop"), ("OCR", "ocr"), ("HIGHRES", "hi")):
        y = np.array([np.mean(s[k] - s["stop"]) for s in states])
        heads[f] = head_factory(winner["head"]).fit(scaler.transform(X), y)
    # freeze lambda(B) on the TextVQA dev subset (in-sample, disclosed)
    tv_idx = [i for i, s in enumerate(states) if s["dataset"] == "textvqa"]
    s_tv = [states[i] for i in tv_idx]
    dh = {f: heads[f].predict(scaler.transform(X))[tv_idx] for f in FAMILIES}
    hi_tok = float(np.mean([s["cost_vtok"]["HIGHRES"] for s in s_tv]))
    lam_star = {}
    for frac in BUDGET_FRACS:
        B = frac * hi_tok
        best = (0.0, -1.0)
        for lam in LAM_TOK:
            ev = tsv.policy_value(s_tv, dh, lam)
            if ev["vtok"] <= B + 1e-9 and ev["acc"] > best[1]:
                best = (float(lam), ev["acc"])
        lam_star[str(frac)] = {"lam": best[0], "dev_acc": round(best[1], 4)}
        print("lambda*", frac, best)
    blob = {"heads": heads, "scaler": scaler,
            "feature_set": winner["features"], "head_name": winner["head"],
            "conf_sorted_mean": np.sort([s["mean_logprob"] for s in states]),
            "lam_tok": LAM_TOK, "frozen_lambda_textvqa": lam_star,
            "embedding": "mean-pooled last hidden layer of the STOP-time "
                         "input (extract_hidden_states.py)",
            "train": f"gate0 dev n={len(states)}", "seed": SEED,
            "candidate_table": table, "v2_reference": v2ref}
    with open(ROOT / "models/selector_frozen_v3.pkl", "wb") as fh:
        pickle.dump(blob, fh)
    sha = hashlib.sha256((ROOT / "models/selector_frozen_v3.pkl").read_bytes()
                         ).hexdigest()[:16]
    (ROOT / "models/selector_frozen_v3.json").write_text(json.dumps(
        {"artifact": "models/selector_frozen_v3.pkl", "sha256_16": sha,
         "seed": SEED, "winner": winner, "candidate_table": table,
         "v2_reference": v2ref, "frozen_lambda_textvqa": lam_star,
         "selection_rule": "max 5-fold CV TextVQA policy value at 1/2x "
                           "(tie 1/4x); full grid disclosed"}, indent=1))
    print("frozen -> models/selector_frozen_v3.pkl", sha)


if __name__ == "__main__":
    main()
