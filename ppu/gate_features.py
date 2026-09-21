"""Features a gate may read before paying for HIGHRES.

v2.1 removes every standalone confidence term. v2 learned a single global
direction (conf_rank -2.66): "confident, therefore look", which fits VQAv2 —
where confidence is inverted — and destroys TextVQA, where it is not. A gate
with no main effect cannot commit to a global sign; it can only decide, per
answer form, whether this uncertainty is worth believing.

Confidence therefore enters only through interactions:
    conf x yes/no      hedging instead of answering
    conf x numeric     counting, where uncertainty IS perceptual
    conf x other       free-form answers
    conf x length      verbosity-weighted uncertainty
    conf x hedge       explicit "can't tell" language

Deliberately absent: dataset id, image resolution, native tokens. Those name
the domain, and a gate that recognises the domain has learned the wrong thing.
Source is used for pair matching and loss weighting only, never as an input.
"""
from __future__ import annotations

import re

_HEDGE = re.compile(r"\b(maybe|possibly|probably|appears|seems|likely|unclear|"
                    r"cannot|can't|unable|not sure|hard to|difficult|blurry|"
                    r"unknown|unreadable|too small)\b", re.I)
_YESNO = re.compile(r"^\s*(yes|no)\b", re.I)
_NUM = re.compile(r"^\s*[-+]?\d+([.,]\d+)?\s*$")

FEATURE_NAMES = [
    # answer surface
    "ans_chars", "ans_tokens", "ans_words", "is_numeric", "is_yesno", "has_hedge",
    # question surface
    "q_chars", "q_words", "q_is_howmany", "q_is_what", "q_is_isare",
    # confidence, interaction-only (no standalone term by design)
    "conf_x_yesno", "conf_x_numeric", "conf_x_other",
    "conf_x_len", "conf_x_hedge",
]


def answer_type(answer: str) -> str:
    if _NUM.match(answer or ""):
        return "num"
    if _YESNO.match(answer or ""):
        return "yesno"
    return "other"


def extract(state: dict, conf_rank: float) -> list[float]:
    """conf_rank: 0 = most confident, 1 = least confident (within the pool)."""
    a = state.get("answer") or state.get("coarse_answer") or ""
    q = state.get("question", "")
    t = answer_type(a)
    is_num = float(t == "num")
    is_yn = float(t == "yesno")
    is_other = float(t == "other")
    hedge = 1.0 if _HEDGE.search(a) else 0.0
    ans_chars = float(len(a))
    ql = q.lower()
    return [
        ans_chars, float(state.get("answer_len_tokens", 0)), float(len(a.split())),
        is_num, is_yn, hedge,
        float(len(q)), float(len(q.split())),
        1.0 if "how many" in ql else 0.0,
        1.0 if ql.strip().startswith("what") else 0.0,
        1.0 if re.match(r"^\s*(is|are|does|do|was|were)\b", ql) else 0.0,
        conf_rank * is_yn, conf_rank * is_num, conf_rank * is_other,
        conf_rank * min(ans_chars / 20.0, 2.0), conf_rank * hedge,
    ]
