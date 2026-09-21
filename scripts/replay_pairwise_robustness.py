#!/usr/bin/env python3
"""Replay-pairwise robustness of the cross-replay protocol (zero inference;
addendum 2026-09-20). Answers: "is the R1→R2 headroom a rewrite artifact?"

Uses the multi-replay collections (replays 1-2 from the Gate-0 logs, replays
3-5 from logs/mr_{ds}.jsonl; PREREG_MULTI_REPLAY) on the 100-state
outcome-blind subsets, deterministic arms only, strict scores as logged.
Replay 3 ("In the image shown: {q}") is an independent rewrite of replay 2
("Looking at / Based on the image, ..."); MCQ replays are fresh permutations.

For every ORDERED replay pair (r, s), r != s, in 1..5:
  headroom(r→s) = oracle(select on r, score on s) − best fixed arm
                  (arm chosen on r over the other half of the states,
                  scored on s; 2-fold by state-id parity — the same
                  cross-fitting rule as multi_replay_analysis.py)
  stop_agree    = P[ 1(y_STOP^r ≥ .5) == 1(y_STOP^s ≥ .5) ]
  arm_corr      = Pearson corr over (state, arm) of y^r(a) vs y^s(a)
                  (arms exclude STOP); arm_agree = binarized agreement
  ppu_sign      = agreement of sign(U^r(a)) and sign(U^s(a)), U = y(a) − y(STOP),
                  3-way {−,0,+} over all (state, arm); plus among pairs where
                  at least one is non-zero ("decisive"), plus among both non-zero.
10k state-paired bootstrap CIs (seed 20260920) for headroom and arm_corr.

Also reports the same consistency statistics for the K=2 pair (1, 2) on the
larger single-reader TextVQA logs (Gate-0 200, prospective 600, v3 300) and
MME 196, so the R1/R2 consistency numbers rest on n up to 600.
Outputs paper/replay_pairwise_robustness.{json,md}.
"""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260920
N_BOOT = 10_000
MR = [("mme", ["logs/g0_mme200.jsonl", "logs/mr_mme.jsonl"]),
      ("textvqa", ["logs/tv_200.jsonl", "logs/mr_textvqa.jsonl"])]
K2 = [("mme / Gate-0 196", ["logs/g0_mme200.jsonl"]),
      ("textvqa / Gate-0 200", ["logs/tv_200.jsonl"]),
      ("textvqa / prospective 600", ["logs/prospective_textvqa.jsonl"]),
      ("textvqa / v3-prospective 300", ["logs/v3_prospective.jsonl"])]


def det_key(a):
    if a["type"] == "STOP":
        return "STOP"
    if a["type"] == "CROP" and a.get("source") == "grid":
        return f"CROP#{a['slot']}"
    if a["type"] in ("OCR", "HIGHRES"):
        return a["type"]
    return None


def load(paths):
    y = defaultdict(dict)
    for p in paths:
        for line in open(ROOT / p):
            r = json.loads(line)
            if r.get("type") != "rollout" or r.get("dummy"):
                continue
            k = det_key(r["action"])
            if k is None:
                continue
            y[r["sample_id"]].setdefault(r["replay"], {})[k] = float(r["score_after"])
    return y


def common(y, replays):
    out = {}
    for sid, reps in y.items():
        if not all(r in reps for r in replays):
            continue
        keys = set(reps[replays[0]])
        if all(set(reps[r]) == keys for r in replays) and "STOP" in keys:
            out[sid] = {r: reps[r] for r in replays}
    return out


def boot_mean(vals):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
    b = vals[idx].mean(axis=1)
    return float(vals.mean()), [float(np.quantile(b, .025)), float(np.quantile(b, .975))]


def boot_corr(x, y, groups):
    """state-paired bootstrap of Pearson corr over (state, arm) rows"""
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    sids = np.unique(groups)
    by = {s: np.where(groups == s)[0] for s in sids}
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(2000):
        pick = rng.integers(0, len(sids), len(sids))
        idx = np.concatenate([by[sids[i]] for i in pick])
        xs, ys = x[idx], y[idx]
        if xs.std() == 0 or ys.std() == 0:
            continue
        out.append(np.corrcoef(xs, ys)[0, 1])
    r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    return r, [float(np.quantile(out, .025)), float(np.quantile(out, .975))] if out else [np.nan, np.nan]


def pair_stats(data, r, s, with_headroom=True):
    sids = sorted(data)
    gkeys = sorted(set.intersection(*(set(data[x][r]) for x in sids)))
    arms = [k for k in gkeys if k != "STOP"]
    res = {"n": len(sids)}
    if with_headroom:
        half = {0: sids[0::2], 1: sids[1::2]}
        bf = {}
        for fold in (0, 1):
            arm = max(gkeys, key=lambda k: np.mean([data[x][r][k] for x in half[fold]]))
            for x in half[1 - fold]:
                bf[x] = data[x][s][arm]
        orc = {x: data[x][s][max(sorted(data[x][r]), key=lambda k: (data[x][r][k], k))]
               for x in sids}
        naive = {x: max(data[x][r].values()) for x in sids}
        hm, hci = boot_mean([orc[x] - bf[x] for x in sids])
        res.update({"oracle_acc": float(np.mean(list(orc.values()))),
                    "naive_same_replay_acc": float(np.mean(list(naive.values()))),
                    "best_fixed_acc": float(np.mean(list(bf.values()))),
                    "headroom": hm, "headroom_ci": hci})
    # STOP consistency
    st = np.array([[data[x][r]["STOP"], data[x][s]["STOP"]] for x in sids])
    res["stop_acc_r"], res["stop_acc_s"] = float(st[:, 0].mean()), float(st[:, 1].mean())
    res["stop_agree_bin"] = float(np.mean((st[:, 0] >= .5) == (st[:, 1] >= .5)))
    res["stop_agree_exact"] = float(np.mean(np.abs(st[:, 0] - st[:, 1]) < 1e-9))
    # action outcome consistency
    xr, xs_, g, ur, us = [], [], [], [], []
    for x in sids:
        for a in arms:
            xr.append(data[x][r][a]); xs_.append(data[x][s][a]); g.append(x)
            ur.append(data[x][r][a] - data[x][r]["STOP"])
            us.append(data[x][s][a] - data[x][s]["STOP"])
    xr, xs_, ur, us = map(lambda v: np.asarray(v, float), (xr, xs_, ur, us))
    res["arm_corr"], res["arm_corr_ci"] = boot_corr(xr, xs_, g)
    res["arm_agree_bin"] = float(np.mean((xr >= .5) == (xs_ >= .5)))
    sr, ss = np.sign(np.round(ur, 6)), np.sign(np.round(us, 6))
    res["ppu_sign_agree_all"] = float(np.mean(sr == ss))
    dec = (sr != 0) | (ss != 0)
    res["ppu_sign_agree_decisive"] = float(np.mean(sr[dec] == ss[dec])) if dec.any() else None
    both = (sr != 0) & (ss != 0)
    res["ppu_sign_agree_both_nonzero"] = float(np.mean(sr[both] == ss[both])) if both.any() else None
    res["n_arm_pairs"] = int(len(xr)); res["n_decisive"] = int(dec.sum()); res["n_both_nonzero"] = int(both.sum())
    res["ppu_corr"] = float(np.corrcoef(ur, us)[0, 1]) if ur.std() > 0 and us.std() > 0 else None
    return res


def f2(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def single_log(log, tag, replays=(1, 2, 3)):
    """Pairwise stats for one log that carries replays 1..3 (e.g. the 400-state
    stronger-controller set) — the R1/R2/R3 triangle at a larger n."""
    base = common(load([log]), list(replays))
    res = {f"{r}->{s}": pair_stats(base, r, s) for r, s in permutations(replays, 2)}
    md = [f"# Replay triangle on {log} (n={len(base)}; replays {replays})", "",
          "| pair | oracle acc | best fixed | headroom [CI] | STOP agree (bin) | arm corr [CI] | arm agree (bin) | PPU corr | PPU sign agree: all / decisive / both≠0 |",
          "|---|--:|--:|:--|--:|:--|--:|--:|:--|"]
    for k, p in res.items():
        r, s = k.split("->")
        md.append(f"| R{r}→R{s} | {p['oracle_acc']:.3f} | {p['best_fixed_acc']:.3f} | "
                  f"{p['headroom']:+.3f} [{p['headroom_ci'][0]:+.3f}, {p['headroom_ci'][1]:+.3f}] | "
                  f"{p['stop_agree_bin']:.3f} | {p['arm_corr']:.3f} [{p['arm_corr_ci'][0]:.3f}, {p['arm_corr_ci'][1]:.3f}] | "
                  f"{p['arm_agree_bin']:.3f} | {f2(p['ppu_corr'])} | {p['ppu_sign_agree_all']:.3f} / {f2(p['ppu_sign_agree_decisive'])} / {f2(p['ppu_sign_agree_both_nonzero'])} |")
    hs = [p["headroom"] for p in res.values()]
    md += ["", f"Mean headroom over the 6 ordered pairs {np.mean(hs):+.3f} (min {min(hs):+.3f}, max {max(hs):+.3f}).", ""]
    (ROOT / f"paper/replay_pairwise_{tag}.json").write_text(json.dumps(res, indent=1))
    (ROOT / f"paper/replay_pairwise_{tag}.md").write_text("\n".join(md))
    print("\n".join(md))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", help="single log with replays 1-3; writes paper/replay_pairwise_<tag>")
    ap.add_argument("--tag", default="strong")
    args = ap.parse_args()
    if args.log:
        single_log(args.log, args.tag)
        return
    out, md = {}, ["# Replay-pairwise robustness of the cross-replay protocol", "",
                   "Deterministic arms, strict scores. headroom(r→s) = oracle(select on",
                   "replay r, score on replay s) − best fixed arm (chosen on r, cross-fitted",
                   "2-fold over states, scored on s). 10k state-paired bootstrap CIs.", ""]
    for ds, paths in MR:
        y = load(paths)
        base = common(y, [1, 2, 3, 4, 5])
        res = {}
        for r, s in permutations(range(1, 6), 2):
            res[f"{r}->{s}"] = pair_stats({x: base[x] for x in base}, r, s)
        out[f"{ds}/multi_replay_100"] = res
        n = len(base)
        md += [f"## {ds} — 100-state multi-replay subset (n={n}), all ordered pairs", "",
               "Headroom (oracle − best fixed) [95% CI]; rows = selection replay r, cols = evaluation replay s:", "",
               "| r \\ s | 1 | 2 | 3 | 4 | 5 |", "|---|:--|:--|:--|:--|:--|"]
        for r in range(1, 6):
            cells = []
            for s in range(1, 6):
                if r == s:
                    cells.append(f"(naive {res[f'{r}->{1 if r != 1 else 2}']['naive_same_replay_acc']:.3f})")
                else:
                    p = res[f"{r}->{s}"]
                    cells.append(f"{p['headroom']:+.3f} [{p['headroom_ci'][0]:+.3f}, {p['headroom_ci'][1]:+.3f}]")
            md.append(f"| {r} | " + " | ".join(cells) + " |")
        hs = [res[k]["headroom"] for k in res]
        md += ["", f"Mean off-diagonal headroom {np.mean(hs):+.3f} (min {min(hs):+.3f}, max {max(hs):+.3f}, sd {np.std(hs):.3f}).",
               "", "The R1/R2/R3 triangle (the three rewrites: original, replay-2 rewrite, replay-3 independent rewrite):", "",
               "| pair | oracle acc | best fixed | headroom [CI] | STOP agree (bin) | arm corr [CI] | arm agree (bin) | PPU sign agree: all / decisive / both≠0 |",
               "|---|--:|--:|:--|--:|:--|--:|:--|"]
        for r, s in ((1, 2), (1, 3), (2, 3), (2, 1), (3, 1), (3, 2)):
            p = res[f"{r}->{s}"]
            md.append(f"| R{r}→R{s} | {p['oracle_acc']:.3f} | {p['best_fixed_acc']:.3f} | "
                      f"{p['headroom']:+.3f} [{p['headroom_ci'][0]:+.3f}, {p['headroom_ci'][1]:+.3f}] | "
                      f"{p['stop_agree_bin']:.3f} | {p['arm_corr']:.3f} [{p['arm_corr_ci'][0]:.3f}, {p['arm_corr_ci'][1]:.3f}] | "
                      f"{p['arm_agree_bin']:.3f} | {p['ppu_sign_agree_all']:.3f} / {f2(p['ppu_sign_agree_decisive'])} / {f2(p['ppu_sign_agree_both_nonzero'])} |")
        md += ["", "Consistency for every unordered pair (symmetric statistics):", "",
               "| pair | STOP agree (bin) | STOP agree (exact) | arm corr | arm agree (bin) | PPU corr | PPU sign: all / decisive / both≠0 |",
               "|---|--:|--:|--:|--:|--:|:--|"]
        for r in range(1, 6):
            for s in range(r + 1, 6):
                p = res[f"{r}->{s}"]
                md.append(f"| R{r}–R{s} | {p['stop_agree_bin']:.3f} | {p['stop_agree_exact']:.3f} | {p['arm_corr']:.3f} | "
                          f"{p['arm_agree_bin']:.3f} | {f2(p['ppu_corr'])} | {p['ppu_sign_agree_all']:.3f} / "
                          f"{f2(p['ppu_sign_agree_decisive'])} / {f2(p['ppu_sign_agree_both_nonzero'])} |")
        md.append("")
    md += ["## K=2 (R1, R2) consistency on the larger single-reader logs", "",
           "| log | n | STOP acc R1 / R2 | STOP agree (bin) | arm corr [CI] | arm agree (bin) | PPU corr | PPU sign: all / decisive / both≠0 | headroom R1→R2 [CI] | headroom R2→R1 [CI] |",
           "|---|--:|:--|--:|:--|--:|--:|:--|:--|:--|"]
    for name, paths in K2:
        base = common(load(paths), [1, 2])
        p12 = pair_stats(base, 1, 2); p21 = pair_stats(base, 2, 1)
        out[f"k2/{name}"] = {"1->2": p12, "2->1": p21}
        md.append(f"| {name} | {p12['n']} | {p12['stop_acc_r']:.3f} / {p12['stop_acc_s']:.3f} | {p12['stop_agree_bin']:.3f} | "
                  f"{p12['arm_corr']:.3f} [{p12['arm_corr_ci'][0]:.3f}, {p12['arm_corr_ci'][1]:.3f}] | {p12['arm_agree_bin']:.3f} | "
                  f"{f2(p12['ppu_corr'])} | {p12['ppu_sign_agree_all']:.3f} / {f2(p12['ppu_sign_agree_decisive'])} / {f2(p12['ppu_sign_agree_both_nonzero'])} | "
                  f"{p12['headroom']:+.3f} [{p12['headroom_ci'][0]:+.3f}, {p12['headroom_ci'][1]:+.3f}] | "
                  f"{p21['headroom']:+.3f} [{p21['headroom_ci'][0]:+.3f}, {p21['headroom_ci'][1]:+.3f}] |")
    md.append("")
    (ROOT / "paper/replay_pairwise_robustness.json").write_text(json.dumps(out, indent=1))
    (ROOT / "paper/replay_pairwise_robustness.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
