#!/usr/bin/env python3
"""Hidden-state × nonlinear controllers — train on the pooled TextVQA logs, FREEZE.

Fills the reviewer's remaining cell: strong representation (the reader's own
STOP-time hidden state H, 3584-d mean-pooled last layer, deployable
information only) × strong head (gradient-boosted trees / 2-layer MLP).
Full factorial so every cell is pool-matched:

  feature sets : F0 (16 surface+confidence) | H | H+F0
  head classes : ridge | xgb | mlp
  targets/rule : identical to v2 / strong controllers (strict replay-mean
                 Δ over STOP per family; argmax_f [Δ̂_f − λ_B c_f], STOP = 0)
  pool         : 1,750 TextVQA states = Gate-0 200 + held-out 250 +
                 prospective 600 + v3 300 + strong 400 (all analyzed sets)

Per cell one configuration from a disclosed grid by 5-fold nested-CV policy
value at ½× (tie ¼×); λ(B) per cell from pooled out-of-fold predictions.
Primary controller for the prereg = better of H-xgb / H-mlp by the same rule.
Frozen to models/hidden_controllers_frozen.{pkl,json} BEFORE the fresh
evaluation set is collected (paper/PREREG_HIDDEN_CONTROLLER.md).
"""
from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".pylib"))
sys.path.insert(0, str(ROOT / "scripts"))
import strong_controller_common as C  # noqa: E402
from strong_controller_common import (BUDGET_FRACS, FAMILIES, LAM_TOK,  # noqa: E402
                                      features, load_states, make_rank,
                                      policy_from_dhat, policy_value,
                                      precompute, targets)

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

SEED = 20260921
N_FOLDS = 5
POOL = dict(C.POOL); POOL["strong"] = ["logs/strong_textvqa.jsonl"]
HNPZ = {"gate0": "data/hidden/dev_textvqa.npz", "heldout": "data/hidden/heldout_textvqa.npz",
        "prospective": "data/hidden/prospective_textvqa.npz", "v3": "data/hidden/v3_prospective.npz",
        "strong": "data/hidden/strong_textvqa.npz"}
FEATSETS = ("F0", "H", "H+F0")
GRID = {
    "ridge": {"F0": [{"alpha": a} for a in (0.3, 1.0, 3.0, 10.0)],
              "H": [{"alpha": a} for a in (3.0, 30.0, 300.0, 3000.0)]},
    "xgb": {"*": [{"max_depth": d, "n_estimators": n, "learning_rate": 0.05, "subsample": 0.8,
                   "colsample_bytree": 0.8, "min_child_weight": 5} for d in (2, 3) for n in (100, 300)]},
    "mlp": {"F0": [{"hidden_layer_sizes": h, "alpha": a} for h in ((64,), (64, 64)) for a in (1e-3, 1e-2)],
            "H": [{"hidden_layer_sizes": h, "alpha": a} for h in ((64,), (64, 64)) for a in (1e-2, 1e-1)]},
}


def grid_for(cls, fs):
    g = GRID[cls]
    return g.get("*") or g["F0" if fs == "F0" else "H"]


def make_head(cls, hp):
    if cls == "ridge":
        return Ridge(alpha=hp["alpha"], random_state=0)
    if cls == "xgb":
        return XGBRegressor(random_state=0, n_jobs=8, verbosity=0, tree_method="hist", **hp)
    if cls == "mlp":
        return MLPRegressor(max_iter=3000, early_stopping=True, validation_fraction=0.15,
                            n_iter_no_change=30, random_state=0, **hp)
    raise ValueError(cls)


def load_H():
    Hmap = {}
    for src, f in HNPZ.items():
        z = np.load(ROOT / f, allow_pickle=True)
        for sid, v in zip(z["ids"], z["H"]):
            Hmap[str(sid)] = v.astype(np.float32)
    return Hmap


def build_X(states, fs, rank, Hmap):
    F0 = features(states, rank)
    if fs == "F0":
        return F0
    H = np.stack([Hmap[s["sid"]] for s in states])
    return H if fs == "H" else np.hstack([H, F0])


def folds(n, seed=SEED):
    rng = np.random.default_rng(seed); a = np.arange(n) % N_FOLDS; rng.shuffle(a); return a


def oof_predict(cls, hp, X, Y, assign):
    dh = {f: np.zeros(len(X)) for f in FAMILIES}
    for k in range(N_FOLDS):
        tr, te = np.where(assign != k)[0], np.where(assign == k)[0]
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for f in FAMILIES:
            dh[f][te] = make_head(cls, hp).fit(Xtr, Y[f][tr]).predict(Xte)
    return dh


def lam_for_budget(states, dhat, B):
    best_lam, best_v = 0.0, -1.0
    for lam in LAM_TOK:
        acc, cost = policy_value(states, policy_from_dhat(states, dhat, lam))
        if cost.mean() <= B + 1e-9 and acc.mean() > best_v:
            best_lam, best_v = lam, acc.mean()
    return best_lam, best_v


def cv_policy_value(states, dh, assign, hi_tok):
    out = {}
    for frac in BUDGET_FRACS:
        B = frac * hi_tok; accs, costs = [], []
        for k in range(N_FOLDS):
            tr, te = np.where(assign != k)[0], np.where(assign == k)[0]
            lam, _ = lam_for_budget([states[i] for i in tr], {f: dh[f][tr] for f in FAMILIES}, B)
            a, c = policy_value([states[i] for i in te],
                                policy_from_dhat([states[i] for i in te], {f: dh[f][te] for f in FAMILIES}, lam))
            accs.append(a); costs.append(c)
        out[frac] = {"acc": float(np.concatenate(accs).mean()), "vtok": float(np.concatenate(costs).mean())}
    return out


def main():
    pool = []
    for src, logs in POOL.items():
        ss, dropped = load_states(logs, source=src)
        print(f"  pool[{src}] {len(ss)} states (dropped {dropped})", flush=True)
        pool += ss
    assert len({s["sid"] for s in pool}) == len(pool)
    precompute(pool)
    Hmap = load_H()
    missing = [s["sid"] for s in pool if s["sid"] not in Hmap]
    assert not missing, f"{len(missing)} pool states without hidden state, e.g. {missing[:3]}"
    n = len(pool)
    rank = make_rank(np.sort([s["mean_logprob"] for s in pool]))
    Y = targets(pool, "strict")
    hi_tok = float(np.mean([s["vtok"]["HIGHRES"] for s in pool]))
    assign = folds(n)
    print(f"pool n={n}, always-HIGHRES {hi_tok:.1f} tok, H dim {next(iter(Hmap.values())).shape[0]}", flush=True)

    Xs = {fs: build_X(pool, fs, rank, Hmap) for fs in FEATSETS}
    table, oofs = [], {}
    for fs in FEATSETS:
        for cls in GRID:
            for hp in grid_for(cls, fs):
                t0 = time.time()
                dh = oof_predict(cls, hp, Xs[fs], Y, assign)
                cv = cv_policy_value(pool, dh, assign, hi_tok)
                row = {"features": fs, "class": cls, "hp": hp,
                       **{f"cv_acc_{f}x": round(cv[f]["acc"], 4) for f in BUDGET_FRACS},
                       **{f"cv_vtok_{f}x": round(cv[f]["vtok"], 1) for f in BUDGET_FRACS},
                       "fit_s": round(time.time() - t0, 1)}
                table.append(row); oofs[(fs, cls, json.dumps(hp, sort_keys=True))] = dh
                print(row, flush=True)
    key, tie = "cv_acc_0.5x", "cv_acc_0.25x"
    winners = {}
    for fs in FEATSETS:
        for cls in GRID:
            winners[f"{fs}/{cls}"] = max([r for r in table if r["features"] == fs and r["class"] == cls],
                                         key=lambda r: (r[key], r[tie]))
    primary = max([winners["H/xgb"], winners["H/mlp"]], key=lambda r: (r[key], r[tie]))
    print("\nWINNERS:", json.dumps(winners, indent=1)); print("primary:", "H/" + primary["class"])

    frozen = {"heads": {}, "scalers": {}, "lam": {}, "hp": {}, "hi_tok": hi_tok, "n_train": n,
              "conf_sorted": np.sort([s["mean_logprob"] for s in pool]),
              "train_sources": sorted({s["source"] for s in pool}), "seed": SEED,
              "primary": "H/" + primary["class"], "hidden_npz": HNPZ}
    lam_json = {}
    for cell, w in winners.items():
        fs, cls = cell.split("/")
        sc = StandardScaler().fit(Xs[fs]); Xz = sc.transform(Xs[fs])
        frozen["scalers"][cell] = sc; frozen["hp"][cell] = w["hp"]
        frozen["heads"][cell] = {f: make_head(cls, w["hp"]).fit(Xz, Y[f]) for f in FAMILIES}
        dh = oofs[(fs, cls, json.dumps(w["hp"], sort_keys=True))]
        frozen["lam"][cell] = {}; lam_json[cell] = {}
        for frac in BUDGET_FRACS:
            lam, v = lam_for_budget(pool, dh, frac * hi_tok)
            frozen["lam"][cell][str(frac)] = lam; lam_json[cell][str(frac)] = {"lam": lam, "oof_acc": round(v, 4)}
    pkl = ROOT / "models/hidden_controllers_frozen.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump(frozen, fh)
    sha = hashlib.sha256(pkl.read_bytes()).hexdigest()[:16]
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (ROOT / "models/hidden_controllers_frozen.json").write_text(json.dumps({
        "artifact": str(pkl.relative_to(ROOT)), "sha256_16": sha, "git": git, "seed": SEED,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_train": n,
        "train_sources": frozen["train_sources"], "feature_sets": FEATSETS,
        "hidden": "mean-pooled last-layer hidden state of the STOP-time input (3584-d)",
        "targets": "strict replay-mean Δ over STOP per family", "winners": winners,
        "primary": frozen["primary"], "frozen_lambda_textvqa": lam_json,
        "pool_highres_vtok": hi_tok, "candidate_table": table,
        "selection_rule": "per (features, class) cell: max 5-fold nested-CV policy value at 1/2x (tie 1/4x); "
                          "primary = better of H/xgb, H/mlp by the same rule; lambda(B) per cell on pooled OOF"},
        indent=1))
    print("frozen ->", pkl, sha)


if __name__ == "__main__":
    main()
