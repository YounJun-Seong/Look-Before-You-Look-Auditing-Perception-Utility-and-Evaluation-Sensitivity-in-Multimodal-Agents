#!/usr/bin/env python
"""Determinism check: re-run the same samples and compare, rollout by rollout.

Greedy decoding on a frozen reader should reproduce exactly. It usually does
not, for reasons that have nothing to do with the research question — prefix
caching, batching nondeterminism, a changed image preprocessing path. Those
have to be found now, because Gate 1 measures effects of a few points and any
of them would forge that size of effect.

The second run must use a FRESH generation cache, otherwise it replays our own
stored answers and proves nothing.

    python scripts/run_gate1.py ... --run-id det20_a --cache data/cache_a.sqlite
    python scripts/run_gate1.py ... --run-id det20_b --cache data/cache_b.sqlite
    python scripts/compare_runs.py logs/det20_a.jsonl logs/det20_b.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def read(path: str) -> dict[tuple, dict]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("type") != "rollout":
                continue
            out[(o["sample_id"], o["action"]["key"], o["replay"])] = o
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--show", type=int, default=8, help="how many mismatches to print")
    ap.add_argument("--threshold", type=float, default=0.97,
                    help="answer agreement below this fails the check")
    args = ap.parse_args()

    a, b = read(args.run_a), read(args.run_b)
    shared = sorted(a.keys() & b.keys())
    if not shared:
        print("no overlapping rollouts", file=sys.stderr)
        return 2

    same_answer = same_score = same_pixels = 0
    mismatches = []
    by_family: Counter = Counter()
    for k in shared:
        ra, rb = a[k], b[k]
        ans_ok = ra["answer_after"].strip() == rb["answer_after"].strip()
        same_answer += ans_ok
        same_score += abs(ra["score_after"] - rb["score_after"]) < 1e-9
        same_pixels += [v["sha"] for v in ra.get("views", [])] == \
                       [v["sha"] for v in rb.get("views", [])]
        if not ans_ok:
            by_family[ra["action"]["family"]] += 1
            mismatches.append((k, ra["answer_after"], rb["answer_after"]))

    n = len(shared)
    print(f"=== determinism over {n} shared rollouts ===")
    print(f"identical answer : {same_answer / n:6.2%}  ({same_answer}/{n})")
    print(f"identical score  : {same_score / n:6.2%}")
    print(f"identical pixels : {same_pixels / n:6.2%}   <- preprocessing")
    if mismatches:
        print(f"\nmismatches by action family: {dict(by_family)}")
        for (sid, key, rep), x, y in mismatches[:args.show]:
            print(f"  {sid} {key} r{rep}: {x!r} != {y!r}")

    ok = (same_answer / n) >= args.threshold and same_pixels == n
    print(f"\n{'PASS' if ok else 'FAIL'} (threshold {args.threshold:.0%})")
    if not ok:
        print("Check, in order: prefix caching on the server, image preprocessing, "
              "cache keys, batching nondeterminism.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
