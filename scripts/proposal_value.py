#!/usr/bin/env python
"""Is a question-conditioned crop proposal worth anything over a random one?

Structured-8 mixes 5 detector-free grid boxes with 3 VLM proposals, so a small
Structured-vs-Random gap could mean either "selection does not matter" or "the
grid diluted the pool". This separates them, cardinality-matched, on the
existing logs (user, 2026-08-17):

    VLM-3     vs Random-3   value of question-conditioned proposals
    Grid-5    vs Random-5   value of plain spatial coverage
    VLM-3     vs Grid-3     is conditioning on the question better than a grid
    Struct-8  vs Random-8   the whole system, as already reported

Every arm is cross-fitted: the best action is chosen on R1 and scored on R2.
Where a comparison needs m of k candidates, the value is averaged over ALL
C(k, m) subsets, so no result depends on which particular boxes were picked.
Naive (choose and score on R1) is printed for reference only.
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

from ppu.audit import SampleView, load

RUNS = [("MME-RealWorld-Lite", "logs/g0_mme200.jsonl"),
        ("TextVQA", "logs/tv_200.jsonl")]


def _oracle(keys, r1, r2) -> float:
    usable = [k for k in keys if k in r1 and k in r2]
    if not usable:
        return float("nan")
    best = min(usable, key=lambda k: (-r1[k], k))
    return r2[best]


def _subset_mean(keys, r1, r2, m: int) -> float:
    """Cross-fitted oracle over m of the keys, averaged over every subset."""
    usable = [k for k in keys if k in r1 and k in r2]
    if len(usable) < m:
        return float("nan")
    vals = [_oracle(list(c), r1, r2) for c in combinations(sorted(usable), m)]
    return float(st.mean(vals))


def _naive_subset_mean(keys, r1, m: int) -> float:
    usable = [k for k in keys if k in r1]
    if len(usable) < m:
        return float("nan")
    return float(st.mean([max(r1[k] for k in c)
                          for c in combinations(sorted(usable), m)]))


def paired_ci(a: list[float], b: list[float], n_boot: int = 10000,
              seed: int = 20260817) -> tuple[float, float, float]:
    d = np.array(a) - np.array(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


def sources(v: SampleView, recs_by_sample: dict) -> dict[str, list[str]]:
    """Split the structured crop keys by where the box came from."""
    out = {"vlm_proposal": [], "grid": []}
    for r in recs_by_sample.get(v.sample_id, []):
        if r["action"]["type"] != "CROP":
            continue
        src = r["action"].get("source", "grid")
        k = r["action"]["key"]
        if k not in out.setdefault(src, []):
            out[src].append(k)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/proposal_value.md")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    md = ["# Does a question-conditioned crop proposal beat a random one?", "",
          "Cardinality-matched, cross-fitted (choose on R1, score on R2). "
          "Where m of k candidates are needed, the value is averaged over every "
          "C(k, m) subset. Paired bootstrap 95% CI over samples.", ""]

    for name, log in RUNS:
        if not Path(log).exists():
            continue
        recs = [json.loads(l) for l in open(log, encoding="utf-8") if l.strip()]
        recs = [r for r in recs if r.get("type") == "rollout"]
        by_sample: dict[str, list[dict]] = {}
        for r in recs:
            by_sample.setdefault(r["sample_id"], []).append(r)
        views = load(recs)

        rows = []
        prop_counts: Counter = Counter()
        for sid, v in views.items():
            if not (v.dummy_r1 and v.dummy_r2):
                continue
            src = sources(v, by_sample)
            vlm, grid = src.get("vlm_proposal", []), src.get("grid", [])
            prop_counts[len(vlm)] += 1
            rnd = sorted(k for k in v.dummy_r1 if k.startswith("CROP_RANDOM#"))
            if len(vlm) < 3 or len(grid) < 3 or len(rnd) < 8:
                continue
            rows.append({
                "sid": sid,
                "vlm3": _subset_mean(vlm, v.r1, v.r2, 3),
                "grid3": _subset_mean(grid, v.r1, v.r2, 3),
                "grid5": _subset_mean(grid, v.r1, v.r2, 5),
                "rnd3": _subset_mean(rnd, v.dummy_r1, v.dummy_r2, 3),
                "rnd5": _subset_mean(rnd, v.dummy_r1, v.dummy_r2, 5),
                "rnd8": _oracle(rnd, v.dummy_r1, v.dummy_r2),
                "struct8": _oracle(vlm + grid, v.r1, v.r2),
                "vlm3_naive": _naive_subset_mean(vlm, v.r1, 3),
                "rnd3_naive": _naive_subset_mean(rnd, v.dummy_r1, 3),
            })
        rows = [r for r in rows if not any(np.isnan(x) for k, x in r.items()
                                           if k != "sid")]
        if not rows:
            md += [f"## {name}", "", "no samples with 3 VLM proposals and both arms", ""]
            continue

        md += [f"## {name} (n = {len(rows)} usable; VLM-proposal counts "
               f"{dict(sorted(prop_counts.items()))})", "",
               "| comparison | arm A | arm B | A − B | 95% CI |",
               "|---|--:|--:|--:|---|"]
        for label, a, b in [("VLM-3 vs Random-3", "vlm3", "rnd3"),
                            ("Grid-5 vs Random-5", "grid5", "rnd5"),
                            ("VLM-3 vs Grid-3", "vlm3", "grid3"),
                            ("Structured-8 vs Random-8", "struct8", "rnd8")]:
            xa = [r[a] for r in rows]
            xb = [r[b] for r in rows]
            d, lo, hi = paired_ci(xa, xb, args.n_boot)
            md.append(f"| {label} | {100 * st.mean(xa):.1f} | {100 * st.mean(xb):.1f} | "
                      f"**{100 * d:+.1f}pp** | [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
        d, lo, hi = paired_ci([r["vlm3_naive"] for r in rows],
                              [r["rnd3_naive"] for r in rows], args.n_boot)
        md += ["", f"reference only (naive, chosen and scored on R1): "
               f"VLM-3 − Random-3 = {100 * d:+.1f}pp [{100 * lo:+.1f}, {100 * hi:+.1f}]",
               ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
