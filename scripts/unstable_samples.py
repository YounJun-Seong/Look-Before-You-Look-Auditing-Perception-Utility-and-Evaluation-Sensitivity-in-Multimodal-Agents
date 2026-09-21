#!/usr/bin/env python
"""Find samples whose *strict label* is not reproducible across passes.

`compare_runs.py` asks an engineering question — is the pipeline deterministic
enough to keep going. This asks the research question: does the solved /
not-solved judgment that every Gate 1 statistic is built on survive a re-run?

A rollout whose answer string changes but still parses to the same choice is
harmless. A rollout that flips `strict(action) = 1 -> 0` is not: diversity,
exclusive-solve, harm and the decisive set are all defined on that bit, so one
flip silently moves a sample between categories.

Any sample with at least one flip is reported as unstable, with per-action
detail, so the Gate 0 audit can report it separately or drop it.

    python scripts/unstable_samples.py logs/g0_det20_a.jsonl logs/g0_det20_b.jsonl \
        --out data/unstable_samples.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppu.audit import THRESHOLD


def read(path: str) -> dict[tuple, dict]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("type") == "rollout":
                out[(o["sample_id"], o["action"]["key"], o["replay"])] = o
    return out


def strict_labels(rows: dict[tuple, dict]) -> dict[tuple, int]:
    """(sample, action) -> 1 when the action solves it under BOTH replays."""
    per: dict[tuple, dict[int, float]] = defaultdict(dict)
    for (sid, key, rep), r in rows.items():
        per[(sid, key)][rep] = r["score_after"]
    return {k: int(all(v >= THRESHOLD for v in d.values()) and len(d) == 2)
            for k, d in per.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--out", default="data/unstable_samples.json")
    args = ap.parse_args()

    a, b = read(args.run_a), read(args.run_b)
    la, lb = strict_labels(a), strict_labels(b)
    shared = sorted(la.keys() & lb.keys())
    if not shared:
        print("no overlapping (sample, action) pairs", file=sys.stderr)
        return 2

    flips: dict[str, list[dict]] = defaultdict(list)
    answer_diffs = 0
    for key in shared:
        sid, action = key
        if la[key] != lb[key]:
            flips[sid].append({"action": action, "pass_a": la[key], "pass_b": lb[key]})
        for rep in (1, 2):
            ra, rb = a.get((sid, action, rep)), b.get((sid, action, rep))
            if ra and rb and ra["answer_after"].strip() != rb["answer_after"].strip():
                answer_diffs += 1

    samples = sorted({k[0] for k in shared})
    payload = {
        "run_a": args.run_a,
        "run_b": args.run_b,
        "n_samples": len(samples),
        "n_action_pairs": len(shared),
        "n_label_flips": sum(len(v) for v in flips.values()),
        "n_unstable_samples": len(flips),
        "unstable_sample_ids": sorted(flips),
        "detail": flips,
        "answer_string_diffs": answer_diffs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"=== strict-label reproducibility over {len(samples)} samples, "
          f"{len(shared)} (sample, action) pairs ===")
    print(f"answer strings differing        : {answer_diffs}")
    print(f"strict-label flips              : {payload['n_label_flips']}")
    print(f"samples with >= 1 flip          : {len(flips)} "
          f"({len(flips) / len(samples):.1%})")
    for sid, items in list(flips.items())[:10]:
        print(f"  {sid}: " + ", ".join(
            f"{i['action']} {i['pass_a']}->{i['pass_b']}" for i in items))
    print(f"\n-> {args.out}")
    print("Zero flips is the target. Any flip means the strict label is not a "
          "property of the sample, and those samples must be reported "
          "separately in the Gate 0 audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
