"""Scoring rules, frozen before any rollout exists.

Gate 1 measures gains and harms of a few points. A noisy judge would drown
exactly the signal we are trying to detect, so there is NO LLM judge here
(SPEC §5.3, user directive 2026-08-16). Every metric is deterministic and
recomputable offline from `raw_generation`.

    vqa_acc : TextVQA / VQAv2 official accuracy, min(#matches/3, 1)
    anls    : DocVQA / InfographicVQA average normalized Levenshtein similarity
    exact   : GQA, normalized exact match
    mcq     : MME-RealWorld, single choice letter

Answers are extracted from the generation before scoring; `parse_ok=False`
means we could not find an answer at all, which is scored 0 but reported
separately (a rising parse-failure rate is a pipeline bug, not a model result).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import Sample

# --- VQA answer normalization (official eval, condensed) --------------------

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Im": "I'm", "isnt": "isn't", "itd": "it'd", "itll": "it'll", "its": "it's",
    "lets": "let's", "maam": "ma'am", "mightve": "might've", "mustve": "must've",
    "shant": "shan't", "shed": "she'd", "shes": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "thats": "that's", "theres": "there's", "theyd": "they'd",
    "theyre": "they're", "theyve": "they've", "wasnt": "wasn't", "werent": "weren't",
    "whatre": "what're", "whats": "what's", "wheres": "where's", "whos": "who's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "youd": "you'd",
    "youre": "you're", "youve": "you've",
}
_NUMBERS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[;/\[\]\"{}()=+\\_\-><@`,?!.']")
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_NUM = re.compile(r"(\d)(,)(\d)")


def normalize_answer(text: str) -> str:
    s = text.replace("\n", " ").replace("\t", " ").strip().lower()
    s = _COMMA_NUM.sub(r"\1\3", s)
    s = _PUNCT.sub("", s)
    s = _PERIOD_STRIP.sub("", s)
    toks = [_NUMBERS.get(t, _CONTRACTIONS.get(t, t)) for t in s.split()]
    toks = [t for t in toks if t not in _ARTICLES]
    return " ".join(toks).strip()


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, golds: tuple[str, ...], threshold: float = 0.5) -> float:
    p = pred.strip().lower()
    best = 0.0
    for g in golds:
        g = g.strip().lower()
        denom = max(len(p), len(g))
        if denom == 0:
            sim = 1.0
        else:
            sim = 1.0 - _levenshtein(p, g) / denom
        best = max(best, sim)
    return best if best >= threshold else 0.0


# --- answer extraction ------------------------------------------------------

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
_ANSWER_LEAD = re.compile(r"(?:final\s+)?answer\s*[:\-]\s*(.+)", re.I)
_MCQ_LETTER = re.compile(r"\b([A-E])\b")


@dataclass
class Scored:
    score: float
    answer: str
    parse_ok: bool


def extract_answer(generation: str, sample: Sample) -> tuple[str, bool]:
    """Pull the answer span out of a generation.

    Deliberately forgiving in a fixed order — tagged span, then 'Answer:' lead,
    then the last non-empty line — because a parser that silently prefers one
    surface form would bias whichever action makes the model chatty.
    """
    text = (generation or "").strip()
    if not text:
        return "", False
    span, ok = None, True
    m = _ANSWER_TAG.search(text)
    if m:
        span = m.group(1)
    else:
        m = _ANSWER_LEAD.search(text)
        if m:
            span = m.group(1).splitlines()[0]
        else:
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            span = lines[-1] if lines else ""
            ok = bool(span)
    span = (span or "").strip().strip("*").strip()
    if sample.metric == "mcq":
        letters = _MCQ_LETTER.findall(span.upper()) or _MCQ_LETTER.findall(text.upper())
        if not letters:
            return span, False
        return letters[0], True
    return span, ok and bool(span)


def score_answer(sample: Sample, answer: str) -> float:
    if sample.metric == "mcq":
        return float(answer.strip().upper()[:1] == sample.answers[0].strip().upper()[:1])
    if sample.metric == "anls":
        return anls(answer, sample.answers)
    norm = normalize_answer(answer)
    golds = [normalize_answer(g) for g in sample.answers]
    if sample.metric == "vqa_acc":
        matches = sum(1 for g in golds if g == norm)
        return min(matches / 3.0, 1.0)
    return float(norm in golds)


def evaluate(sample: Sample, generation: str) -> Scored:
    answer, ok = extract_answer(generation, sample)
    return Scored(score=score_answer(sample, answer) if ok else 0.0,
                  answer=answer, parse_ok=ok)


def binarize(score: float, threshold: float = 0.5) -> int:
    """Gate 1 counts (exclusive solve, harm, diversity) are defined on binary
    outcomes; soft VQA/ANLS scores are thresholded at 0.5 (SPEC §5.3)."""
    return int(score >= threshold)
