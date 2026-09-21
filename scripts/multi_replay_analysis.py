#!/usr/bin/env python3
"""Multi-replay robustness analysis (paper/PREREG_MULTI_REPLAY.md estimands).

Joins the Gate 0 logs (replays 1-2) with the mr_{ds} logs (replays 3-5) on
(sample_id, deterministic action key) and computes, for K = 2..5:

  naive      same-replay oracle, averaged over replays 1..K
  cross_1to1 select on R1, score on R2 (the paper's protocol; K-invariant)
  heldout    select on R1, score on mean(R2..RK)
  symmetric  select on r, score on mean of the other replays in 1..K, avg r

each as oracle accuracy and headroom vs the best fixed arm (same subset,
cross-fitted 2-fold by state id parity), with 10k state-paired bootstrap CIs
(seed 20260830). Deterministic arms only; strict scores as logged.

Outputs paper/multi_replay_robustness.{json,md}.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEED = 20260830
N_BOOT = 10_000
DS = [("mme", "logs/g0_mme200.jsonl", "logs/mr_mme.jsonl"),
      ("textvqa", "logs/tv_200.jsonl", "logs/mr_textvqa.jsonl")]


def det_key(a):
    if a["type"] == "STOP":
        return "STOP"
    if a["type"] == "CROP" and a.get("source") == "grid":
        # slot identifies the grid position; exact boxes vary with image
        # aspect ratio but are identical across replays within a state
        return f"CROP#{a['slot']}"
    if a["type"] in ("OCR", "HIGHRES"):
        return a["type"]
    return None


def load(paths):
    y = defaultdict(dict)  # sid -> replay -> {key: score}
    for p in paths:
        for line in open(ROOT / p):
            r = json.loads(line)
            if r.get("type") != "rollout":
                continue
            k = det_key(r["action"])
            if k is None:
                continue
            y[r["sample_id"]].setdefault(r["replay"], {})[k] = r["score_after"]
    return y


def common(y, replays):
    """states having every replay with an identical full key set"""
    out = {}
    for sid, reps in y.items():
        if not all(r in reps for r in replays):
            continue
        keys = set(reps[replays[0]])
        if all(set(reps[r]) == keys for r in replays) and "STOP" in keys:
            out[sid] = {r: reps[r] for r in replays}
    return out


def stat(fn, sids, n_boot=N_BOOT):
    """bootstrap over states of a per-state statistic function"""
    vals = np.array([fn(s) for s in sids], dtype=float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    boots = vals[idx].mean(axis=1)
    return float(vals.mean()), [float(np.quantile(boots, q)) for q in (0.025, 0.975)]


def main():
    out = {}
    md = ["# Multi-replay robustness (prereg: paper/PREREG_MULTI_REPLAY.md)", "",
          "Oracle accuracy (and headroom vs cross-fitted best fixed) on the",
          "deterministic arm set, as the number of replays K grows. 100-state",
          "outcome-blind subsets; strict scores; 10k state-paired bootstrap.", ""]
    for ds, g0, mr in DS:
        y = load([g0, mr])
        res = {}
        # reference: K=2 on the full Gate 0 sample (protocol as-published)
        full2 = common(y, [1, 2])
        fs = sorted(full2)

        def _c11(s):
            k = max(sorted(full2[s][1]), key=lambda k: (full2[s][1][k], k))
            return full2[s][2][k]
        ref, ref_ci = stat(_c11, fs)
        md += [f"## {ds}", "",
               f"Reference (full sample, n={len(fs)}): K=2 1→1 oracle "
               f"{ref:.3f} [{ref_ci[0]:.3f}, {ref_ci[1]:.3f}]. "
               "Rows below are the fixed 100-state subset for every K.", "",
               "| K | naive same-replay | 1→1 cross | 1→mean(R2..RK) | symmetric LOO | best fixed |",
               "|---|--:|--:|--:|--:|--:|"]
        base = common(y, [1, 2, 3, 4, 5])  # same states for every K row
        for K in (2, 3, 4, 5):
            reps = list(range(1, K + 1))
            data = {s: {r: base[s][r] for r in reps} for s in base}
            sids = sorted(data)
            if not sids:
                md.append(f"| {K} | — | — | — | — | — | (no complete states)")
                continue
            # fixed-arm vocabulary must exist in every state (grid slot count
            # can vary with image shape); per-state oracles use all its arms
            gkeys = sorted(set.intersection(*(set(data[s][1]) for s in sids)))

            # cross-fitted best fixed: pick arm on even-index states, apply to odd (and vice versa)
            half = {0: sids[0::2], 1: sids[1::2]}
            bf_val = {}
            for fold in (0, 1):
                train = half[fold]
                arm = max(gkeys, key=lambda k: np.mean(
                    [data[s][r][k] for s in train for r in reps]))
                for s in half[1 - fold]:
                    bf_val[s] = np.mean([data[s][r][arm] for r in reps])
            bf, bf_ci = stat(lambda s: bf_val[s], sids)

            def naive(s):
                return np.mean([max(data[s][r].values()) for r in reps])

            def sel(s, r):
                return max(sorted(data[s][r]), key=lambda k: (data[s][r][k], k))

            def cross11(s):
                return data[s][2][sel(s, 1)]

            def heldout(s):
                k = sel(s, 1)
                return np.mean([data[s][r][k] for r in reps[1:]])

            def sym(s):
                vals = []
                for r in reps:
                    k = sel(s, r)
                    vals.append(np.mean([data[s][q][k] for q in reps if q != r]))
                return np.mean(vals)

            row = {"n": len(sids)}
            for name, fn in (("naive", naive), ("cross_1to1", cross11),
                             ("heldout_mean", heldout), ("symmetric", sym)):
                m, ci = stat(fn, sids)
                row[name] = {"acc": m, "ci": ci,
                             "headroom_vs_bestfixed": m - bf}
            row["best_fixed"] = {"acc": bf, "ci": bf_ci}
            res[K] = row
            P = lambda d: f"{d['acc']:.3f}"
            md.append(f"| {K} (n={len(sids)}) | {P(row['naive'])} | "
                      f"{P(row['cross_1to1'])} | {P(row['heldout_mean'])} | "
                      f"{P(row['symmetric'])} | {P(row['best_fixed'])} |")
        md += ["", "Headroom (oracle − best fixed) with 95% CI of the oracle:", ""]
        md += ["| K | 1→1 | 1→mean | symmetric |", "|---|--:|--:|--:|"]
        for K, row in res.items():
            f = lambda nm: (f"+{row[nm]['headroom_vs_bestfixed']:.3f} "
                            f"[{row[nm]['ci'][0]:.3f}, {row[nm]['ci'][1]:.3f}]")
            md.append(f"| {K} | {f('cross_1to1')} | {f('heldout_mean')} | {f('symmetric')} |")
        md.append("")
        out[ds] = res
    (ROOT / "paper/multi_replay_robustness.json").write_text(json.dumps(out, indent=1))
    (ROOT / "paper/multi_replay_robustness.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
