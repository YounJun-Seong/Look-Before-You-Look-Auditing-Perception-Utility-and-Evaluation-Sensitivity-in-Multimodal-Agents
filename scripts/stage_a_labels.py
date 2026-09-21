#!/usr/bin/env python
"""Stage A0 labels: one row per coarse STATE, not per rollout.

The independent unit is the state. The many actions and the two replays exist
to *measure* that state's outcome, so they collapse into a single label:

    LOOK       X beats STOP under BOTH replays      -> positive
    STOP       X never beats STOP under either      -> negative
    UNCERTAIN  the replays disagree                 -> excluded from training

UNCERTAIN is not forced into 0/1. Gate 0 measured that the teacher signal is
not deterministic (a bare re-ask moved TextVQA by +11pp); pushing those states
into binary supervision would train the gate on model noise.

Only features visible before paying for X are carried over. Dataset and task
metadata are deliberately excluded from the feature set — they are kept in the
row for slicing the results, never for fitting.

    python scripts/stage_a_labels.py --look HIGHRES --out data/stage_a_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ppu.audit import THRESHOLD
from ppu.datasets import load_manifest

RUNS = [("mme", "logs/g0_mme200.jsonl", "data/manifests/mme_realworld_lite.jsonl"),
        ("textvqa", "logs/tv_200.jsonl", "data/manifests/textvqa_val.jsonl")]


def look_key(rows: list[dict], look: str) -> str | None:
    """Resolve the LOOK action to a concrete logged key."""
    if look == "HIGHRES":
        return "HIGHRES#0"
    if look == "OCR":
        return "OCR#0"
    if look.startswith("CROP"):
        slot = look.replace("CROP", "").strip() or "0"
        for r in rows:
            if r["action"]["type"] == "CROP" and str(r["action"]["slot"]) == slot:
                return r["action"]["key"]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", default="HIGHRES",
                    help="the single action a LOOK decision executes")
    ap.add_argument("--out", default="data/stage_a_labels.jsonl")
    args = ap.parse_args()

    out_rows = []
    for ds, log, manifest in RUNS:
        if not Path(log).exists():
            continue
        recs = [json.loads(l) for l in open(log, encoding="utf-8") if l.strip()]
        recs = [r for r in recs if r.get("type") == "rollout" and not r.get("dummy")]
        meta = {s.sample_id: s for s in load_manifest(manifest)}

        by_state: dict[str, dict] = {}
        for r in recs:
            by_state.setdefault(r["sample_id"], []).append(r)

        for sid, rows in by_state.items():
            key = look_key(rows, args.look)
            get = lambda k, rep: next(
                (x["score_after"] for x in rows
                 if x["action"]["key"] == k and x["replay"] == rep), None)
            s1, s2 = get("STOP#0", 1), get("STOP#0", 2)
            x1, x2 = (get(key, 1), get(key, 2)) if key else (None, None)
            if None in (s1, s2, x1, x2):
                continue
            b = lambda v: v >= THRESHOLD
            gain1, gain2 = b(x1) and not b(s1), b(x2) and not b(s2)
            worse1, worse2 = b(s1) and not b(x1), b(s2) and not b(x2)
            if gain1 and gain2:
                label = "LOOK"
            elif not gain1 and not gain2:
                label = "STOP"
            else:
                label = "UNCERTAIN"
            harmful = worse1 and worse2

            stop_row = next(x for x in rows
                            if x["action"]["key"] == "STOP#0" and x["replay"] == 1)
            s = meta.get(sid)
            out_rows.append({
                "state_id": sid, "dataset": ds, "look_action": args.look,
                "label": label, "harmful": harmful,
                # features available before paying for X
                "question": stop_row["question"],
                "coarse_answer": stop_row["answer_after"],
                "answer_len_chars": len(stop_row["answer_after"]),
                "answer_len_tokens": stop_row["cost"]["tokens_out"],
                "coarse_visual_tokens": stop_row["views"][0]["tokens"],
                "native_tokens": stop_row["native_tokens"],
                "budget_ratio": round(stop_row["native_tokens"]
                                      / max(stop_row["views"][0]["tokens"], 1), 2),
                "question_len_chars": len(stop_row["question"]),
                "image_path": stop_row["image_path"],
                # kept for slicing only, never a model input
                "meta_domain": (s.meta.get("subtask") or s.meta.get("category") or "-")
                               if s else "-",
                "stop_r1": s1, "stop_r2": s2, "look_r1": x1, "look_r2": x2,
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in out_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"LOOK action = {args.look};  {len(out_rows)} states -> {args.out}\n")
    print(f"{'dataset':10s} {'n':>5s} {'LOOK':>8s} {'STOP':>8s} {'UNCERTAIN':>10s} {'harmful':>8s}")
    for ds in ["mme", "textvqa", "ALL"]:
        rs = [r for r in out_rows if ds == "ALL" or r["dataset"] == ds]
        if not rs:
            continue
        c = Counter(r["label"] for r in rs)
        n = len(rs)
        print(f"{ds:10s} {n:5d} "
              f"{c['LOOK']:4d} ({100 * c['LOOK'] / n:4.1f}%) "
              f"{c['STOP']:4d} ({100 * c['STOP'] / n:4.1f}%) "
              f"{c['UNCERTAIN']:4d} ({100 * c['UNCERTAIN'] / n:4.1f}%) "
              f"{sum(1 for r in rs if r['harmful']):4d} "
              f"({100 * sum(1 for r in rs if r['harmful']) / n:4.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
