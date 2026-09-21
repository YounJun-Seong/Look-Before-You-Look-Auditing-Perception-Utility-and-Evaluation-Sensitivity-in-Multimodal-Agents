#!/usr/bin/env python
"""Produce the Gate 1 Audit Table from rollout logs.

    python scripts/analyze_gate1.py logs/g1_*.jsonl --out paper/gate1_audit.md

Reads only the logs. The verdict comes from the pre-registered thresholds in
SPEC_GATE01 §9 and is printed whether or not we like it.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppu.audit import (audit, bottleneck, by_dataset, by_meta, load,
                       render_markdown, verdict)
from ppu.datasets import load_manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="rollout JSONL files (globs ok)")
    ap.add_argument("--out", default="paper/gate1_audit.md")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--manifest", default="",
                    help="manifest to join for per-stratum breakdowns")
    ap.add_argument("--strata", nargs="*", default=["subtask", "category"],
                    help="manifest meta keys to break the audit down by")
    args = ap.parse_args()

    paths: list[str] = []
    for pattern in args.logs:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    records = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("type") == "rollout":
                    records.append(obj)
    if not records:
        print("no rollout records found", file=sys.stderr)
        return 2

    views = load(records)
    pooled = audit(views, "pooled", n_boot=args.n_boot)
    per_ds = by_dataset(views, n_boot=max(1000, args.n_boot // 5))
    verd = verdict(pooled, per_ds)
    bottle = bottleneck(views)

    strata_tables = {}
    if args.manifest:
        meta = {s.sample_id: s.meta for s in load_manifest(args.manifest)}
        for key in args.strata:
            strata_tables[key] = by_meta(views, meta, key)

    md = render_markdown(pooled, per_ds, verd, bottle)
    for key, tables in strata_tables.items():
        md += f"\n\n## Breakdown by {key}\n\n"
        md += ("| " + key + " | n | coarse | full | oracle cf | dummy | H net | "
               "best STOP | CROP | OCR | HIGHRES |\n|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n")
        for t in tables:
            if t.get("note"):
                md += f"| {t['label']} | {t['n']} | (too small) | | | | | | | | |\n"
                continue
            a, h, d = t["accuracy"], t["headroom"], t["diversity"]["distribution"]
            f = lambda x: "—" if x is None else f"{100 * x:.1f}"
            md += (f"| {t['label']} | {t['n']} | {f(a['coarse_stop_r2'])} | "
                   f"{f(a['full_image_r2'])} | {f(a['oracle_cf'])} | "
                   f"{f(a['oracle_dummy_cf'])} | {f(h['H_net'])} | "
                   + " | ".join(f(d[x]) for x in ("STOP", "CROP", "OCR", "HIGHRES"))
                   + " |\n")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"pooled": pooled, "per_dataset": per_ds,
                        "verdict": verd, "bottleneck": bottle,
                        "strata": strata_tables},
                       indent=2), encoding="utf-8")

    print(md)
    print(f"\n[{len(records)} rollouts, {len(views)} samples] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
