#!/usr/bin/env python3
"""Blinded human-semantic utility audit — annotation package builder.

Exp. 2 of the 2026-08-28 revision plan: hold the model, prompt, generations,
and confidence fixed; change ONLY the evaluator. A human judges each unique
(state, answer) pair against the gold references, blinded to which arm
(STOP/HIGHRES) and replay produced it, in a globally shuffled order. The
analysis (scripts/human_audit_analysis.py) then rebuilds PPU labels from the
human scores and re-evaluates the FROZEN original-prompt confidence ranking
against the human-defined target.

Outputs:
  data/human_audit/items_{vqav2,textvqa}.jsonl   private mapping (do not open
                                                 while annotating)
  paper/human_audit_sheet_{vqav2,textvqa}.tsv    blinded sheets: fill `verdict`
                                                 with C / P / I / A
                                                 (Correct / Partial / Incorrect
                                                 / Ambiguous-reference), and
                                                 optionally `note`
  paper/PREREG_HUMAN_AUDIT.md is written separately and fixes the analysis
  before any verdict is entered.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED = 20260828
LOGS = {"vqav2": "logs/fresh_vqav2_orig.jsonl",
        "textvqa": "logs/fresh_textvqa_orig.jsonl"}


def norm(a: str) -> str:
    return " ".join((a or "").lower().split())


def gold_summary(golds) -> str:
    return ", ".join(f"{a} x{c}" for a, c in
                     Counter(norm(g) for g in golds).most_common())


def main():
    (ROOT / "data/human_audit").mkdir(parents=True, exist_ok=True)
    for ds, log in LOGS.items():
        items = {}
        for line in open(ROOT / log):
            r = json.loads(line)
            if r.get("type") != "rollout" or r["action"]["type"] not in ("STOP", "HIGHRES"):
                continue
            sid = r["sample_id"]
            key = (sid, norm(r["answer_after"]))
            it = items.setdefault(key, {
                "sample_id": sid, "question": r["question"], "gold": r["gold"],
                "answer_display": (r["answer_after"] or "").strip(),
                "answer_norm": norm(r["answer_after"]),
                "strict_score": float(r["score_after"]), "sources": [],
            })
            it["sources"].append({"arm": r["action"]["type"], "replay": r["replay"],
                                  "score_after": float(r["score_after"])})
            # identical normalized answers can score differently only if the
            # scorer is non-deterministic; record any disagreement
            if abs(float(r["score_after"]) - it["strict_score"]) > 1e-9:
                it["strict_score_disagreement"] = True

        rows = list(items.values())
        for it in rows:
            h = hashlib.sha256(f"{SEED}|{ds}|{it['sample_id']}|{it['answer_norm']}"
                               .encode()).hexdigest()[:10]
            it["item_id"] = f"{ds[:1]}{h}"
        rng = random.Random(f"{SEED}-{ds}")
        rng.shuffle(rows)

        with open(ROOT / f"data/human_audit/items_{ds}.jsonl", "w") as fh:
            for it in rows:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")

        sheet = ROOT / f"paper/human_audit_sheet_{ds}.tsv"
        with open(sheet, "w") as fh:
            fh.write("item_id\tquestion\tgold_references\tanswer\tverdict\tnote\n")
            for it in rows:
                q = it["question"].replace("\t", " ").replace("\n", " ")
                g = gold_summary(it["gold"]).replace("\t", " ")
                a = it["answer_display"].replace("\t", " ").replace("\n", " / ")
                fh.write(f"{it['item_id']}\t{q}\t{g}\t{a}\t\t\n")
        print(f"{ds}: {len(rows)} blinded items -> {sheet.name} "
              f"(states {len({it['sample_id'] for it in rows})})")


if __name__ == "__main__":
    main()
