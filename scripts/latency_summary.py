#!/usr/bin/env python3
"""Latency-axis summary at the matched-budget read-offs (post-hoc, no compute).

The token-budget tables optimize/compare on visual tokens; this pulls the
already-logged mean added latency (generation + OCR tool time) of the SAME
operating points, so every token-axis claim has its latency counterpart next
to it. Numbers come verbatim from paper/heldout_multiaction.json.

Outputs paper/latency_summary.md.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = json.load(open(ROOT / "paper/heldout_multiaction.json"))["datasets"]

md = ["# Latency at the token-matched operating points (held-out, strict)", "",
      "Mean added latency per state (generation + OCR tool time, seconds) of",
      "the operating points reported in the token-budget tables. Same frozen",
      "numbers, different cost axis; the two axes are never summed.", ""]
for ds, v in D.items():
    s = v["scorers"]["strict"]
    rows = s["rows"]
    md += [f"## {ds}", "",
           "| policy @ budget | acc | vtok | latency (s) |", "|---|--:|--:|--:|",
           f"| always STOP | {rows['always_STOP']['acc']:.3f} | 0 | 0.00 |",
           f"| always HIGHRES | {rows['always_HIGHRES']['acc']:.3f} | "
           f"{rows['always_HIGHRES']['vtok']:.0f} | "
           f"{rows['always_HIGHRES']['lat']:.2f} |",
           f"| always OCR | {rows['always_OCR']['acc']:.3f} | "
           f"{rows['always_OCR']['vtok']:.0f} | {rows['always_OCR']['lat']:.2f} |",
           f"| always CROP | {rows['always_CROP']['acc']:.3f} | "
           f"{rows['always_CROP']['vtok']:.0f} | {rows['always_CROP']['lat']:.2f} |"]
    for bt in s["budget_table"]:
        f = bt["budget_frac"]
        for name in ("confidence", "v22_gate", "multiaction"):
            p = bt[name]
            md.append(f"| {name} @ {f:g}× | {p['acc']:.3f} | {p['vtok']:.0f} "
                      f"| {p['lat']:.2f} |")
    o = rows["oracle_cross_replay"]
    md.append(f"| cross-replay oracle | {o['acc']:.3f} | {o['vtok']:.0f} "
              f"| {o['lat']:.2f} |")
    md.append("")
(ROOT / "paper/latency_summary.md").write_text("\n".join(md))
print("\n".join(md))
