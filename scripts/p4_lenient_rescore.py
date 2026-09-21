#!/usr/bin/env python3
"""Phi-4 lenient rescore (offline, no inference).

Rescores the Phi-4 Gate-0-style logs with the frozen lenient rule of
scripts/gate0_lenient_recheck.py — gold counts as hit if the normalised gold
equals the normalised prediction, equals its first token, or appears as a
whole-word substring. TextVQA keeps the VQA convention min(1, hits/3);
MME (single-gold MCQ) is binary any-hit (the /3 convention would cap a
correct answer at 1/3). Strict reproduction is asserted before rescoring
(binary-exact for MME, vqa_strict for TextVQA; 0 mismatches).

Writes logs/derived_p4_{ds}_lenient.jsonl for the frozen audit + frontier.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppu.evaluator import normalize_answer  # noqa: E402


def hit(g, p, toks):
    g = normalize_answer(g)
    return g != "" and (g == p or (toks and g == toks[0]) or f" {g} " in f" {p} ")


def mcq(pred, gold):
    # evaluator.score_answer: extracted letter, first-char comparison. The
    # lenient relaxation targets open-ended surface forms; for MCQ letters it
    # is a no-op, so lenient == strict by construction.
    return float((pred or "").strip().upper()[:1] == gold.strip().upper()[:1])


def lenient(pred, golds):
    if len(golds) == 1:
        return mcq(pred, golds[0])
    p = normalize_answer(pred or "")
    toks = p.split()
    return min(1.0, sum(hit(g, p, toks) for g in golds) / 3)


def vqa_strict(pred, golds):
    if len(golds) == 1:
        return mcq(pred, golds[0])
    p = normalize_answer(pred or "")
    return min(1.0, sum(normalize_answer(g) == p for g in golds) / 3)


for ds in ("mme", "textvqa"):
    src = ROOT / f"logs/p4_{ds}200.jsonl"
    dst = ROOT / f"logs/derived_p4_{ds}_lenient.jsonl"
    n = mism = 0
    with open(dst, "w") as fh:
        for line in open(src):
            r = json.loads(line)
            if r.get("type") == "rollout" and r.get("parse_ok", True):
                if abs(vqa_strict(r["answer_after"], r["gold"]) - r["score_after"]) > 1e-9:
                    mism += 1
                r["score_after"] = lenient(r["answer_after"], r["gold"])
                if r.get("score_before") is not None and r["action"]["type"] != "STOP":
                    pass  # score_before untouched; audit uses score_after
                n += 1
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    assert mism == 0, f"{ds}: strict not reproduced ({mism})"
    print(f"{ds}: rescored {n} rollouts -> {dst.name} (strict reproduction 0 mismatches)")
