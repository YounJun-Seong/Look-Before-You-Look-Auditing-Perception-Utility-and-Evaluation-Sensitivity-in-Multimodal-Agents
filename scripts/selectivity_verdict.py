#!/usr/bin/env python
"""The decisive comparison: is question-conditioned crop selection worth anything?

Joins the recovered-proposal rollouts (logs/sel_*.jsonl) with the Gate 0 logs
and answers, cardinality-matched per sample and cross-fitted throughout:

    VLM-k vs MatchedRandom-k   question conditioning, size/aspect held constant
    VLM-k vs Grid-k            conditioning vs deterministic coverage
    VLM-k vs FULL              selective local observation vs seeing everything
    VLM-k vs Random-k          (reference: unmatched random boxes)

k is the number of valid proposals for that sample (1..3); samples with k = 0
are excluded here and reported as a proposal-failure rate. Where an arm has
more than k candidates, its value is averaged over every C(n, k) subset so the
result never depends on which particular boxes were compared.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ppu.audit import THRESHOLD, load

RUNS = [("MME-RealWorld-Lite", "logs/sel_mme.jsonl", "logs/g0_mme200.jsonl",
         "logs/sel_mme_proposals.json"),
        ("TextVQA", "logs/sel_textvqa.jsonl", "logs/tv_200.jsonl",
         "logs/sel_textvqa_proposals.json")]


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _oracle(keys, r1, r2):
    usable = [k for k in keys if k in r1 and k in r2]
    if not usable:
        return None
    best = min(usable, key=lambda k: (-r1[k], k))
    return r2[best]


def _subset(keys, r1, r2, m):
    usable = sorted(k for k in keys if k in r1 and k in r2)
    if len(usable) < m:
        return None
    return float(st.mean([_oracle(list(c), r1, r2)
                          for c in combinations(usable, m)]))


def paired(a, b, n_boot=10000, seed=20260817):
    d = np.array(a) - np.array(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(axis=1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/selectivity_verdict.md")
    args = ap.parse_args()

    md = ["# Is question-conditioned crop selection worth anything?", "",
          "Per sample, k = number of valid VLM proposals. Every arm is matched to "
          "that k, chosen on R1 and scored on R2; arms with more than k candidates "
          "are averaged over all C(n, k) subsets. Paired bootstrap 95% CI.", ""]
    pooled: dict[str, list[float]] = {}

    for name, sel_log, g0_log, prop_json in RUNS:
        if not Path(sel_log).exists():
            continue
        sel = load([r for r in read(sel_log) if r.get("type") == "rollout"])
        g0_recs = [r for r in read(g0_log) if r.get("type") == "rollout"]
        g0 = load(g0_recs)
        grid_keys: dict[str, list[str]] = {}
        for r in g0_recs:
            if r["action"]["type"] == "CROP" and r["action"].get("source") == "grid":
                grid_keys.setdefault(r["sample_id"], [])
                if r["action"]["key"] not in grid_keys[r["sample_id"]]:
                    grid_keys[r["sample_id"]].append(r["action"]["key"])
        stats = json.loads(Path(prop_json).read_text())["stats"]

        rows = []
        for sid, v in sel.items():
            g = g0.get(sid)
            if g is None:
                continue
            vlm = sorted(v.r1)
            matched = sorted(v.dummy_r1)
            k = len(vlm)
            if k == 0 or len(matched) < k:
                continue
            rnd8 = sorted(x for x in g.dummy_r1 if x.startswith("CROP_RANDOM#"))
            row = {
                "sid": sid, "k": k,
                "vlm": _oracle(vlm, v.r1, v.r2),
                "matched": _oracle(matched, v.dummy_r1, v.dummy_r2),
                "grid": _subset(grid_keys.get(sid, []), g.r1, g.r2, k),
                "rnd": _subset(rnd8, g.dummy_r1, g.dummy_r2, k) if rnd8 else None,
                "full": g.full_r2,
                "stop": g.r2.get("STOP#0"),
            }
            if None in (row["vlm"], row["matched"], row["grid"], row["full"], row["stop"]):
                continue
            rows.append(row)

        if not rows:
            md += [f"## {name}", "", "no usable samples", ""]
            continue
        tot = sum(stats[f"k{i}"] for i in range(4))
        kdist = Counter(r["k"] for r in rows)
        md += [f"## {name}", "",
               f"proposal failure rate (k = 0): **{100 * stats['k0'] / tot:.1f}%** "
               f"({stats['k0']}/{tot}); usable samples n = {len(rows)}, "
               f"k distribution {dict(sorted(kdist.items()))}", "",
               "| comparison | arm A | arm B | A − B | 95% CI | verdict |",
               "|---|--:|--:|--:|---|---|"]
        for label, a, b in [("VLM-k vs MatchedRandom-k", "vlm", "matched"),
                            ("VLM-k vs Grid-k", "vlm", "grid"),
                            ("VLM-k vs FULL", "vlm", "full"),
                            ("VLM-k vs Random-k (unmatched)", "vlm", "rnd")]:
            xs = [r for r in rows if r[b] is not None]
            if not xs:
                continue
            A = [r[a] for r in xs]
            B = [r[b] for r in xs]
            d, lo, hi = paired(A, B)
            sig = "significant" if lo > 0 else ("negative" if hi < 0 else "n.s.")
            md.append(f"| {label} | {100 * st.mean(A):.1f} | {100 * st.mean(B):.1f} | "
                      f"**{100 * d:+.1f}pp** | [{100 * lo:+.1f}, {100 * hi:+.1f}] | {sig} |")
            pooled.setdefault(label, []).extend(np.array(A) - np.array(B))
        md += ["",
               f"reference: STOP {100 * st.mean([r['stop'] for r in rows]):.1f}, "
               f"FULL {100 * st.mean([r['full'] for r in rows]):.1f}", ""]

        # does the advantage concentrate where the coarse view failed?
        md += ["| subset | n | VLM − MatchedRandom |", "|---|--:|--:|"]
        for flag, lab in ((0, "coarse wrong"), (1, "coarse right")):
            sub = [r for r in rows if int(r["stop"] >= THRESHOLD) == flag]
            if sub:
                d = st.mean([r["vlm"] - r["matched"] for r in sub])
                md.append(f"| {lab} | {len(sub)} | {100 * d:+.1f}pp |")
        md.append("")

    if pooled:
        md += ["## Pooled across datasets", "",
               "| comparison | A − B | 95% CI |", "|---|--:|---|"]
        for label, diffs in pooled.items():
            d = np.array(diffs)
            rng = np.random.default_rng(7)
            idx = rng.integers(0, len(d), size=(10000, len(d)))
            m = d[idx].mean(axis=1)
            lo, hi = np.quantile(m, [0.025, 0.975])
            md.append(f"| {label} | **{100 * d.mean():+.1f}pp** | "
                      f"[{100 * lo:+.1f}, {100 * hi:+.1f}] |")
        md.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
