"""Replay rendering (SPEC §6.1).

Greedy decoding means re-running a request returns the same string, so a
replay cannot measure decoding noise — and decoding noise is not what threatens
this experiment. What threatens it is *luck*: an action that happens to make a
5-way multiple choice land on the right letter, or a phrasing the reader
latches onto. So R2 changes what is asked, never how it is sampled:

    MCQ         : fixed permutation of the answer options
    open-ended  : a fixed, meaning-preserving question rewrite

Every metric in the audit is then computed by choosing on R1 and scoring on R2,
which prices in the winner's curse of taking a max over 11 actions.

The permutation and the rewrite are deterministic functions of the sample id,
so the two replays are reproducible from the manifest alone.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from .schema import Sample


@dataclass
class ServedQuestion:
    text: str  # question as rendered (choices inlined for MCQ)
    choices: tuple[str, ...]  # order actually shown
    gold_letter: str  # correct letter under THIS order ("" for open-ended)


def _permute_choices(sample: Sample, seed: int) -> tuple[tuple[str, ...], str]:
    """Permute options and follow the gold answer to its new letter."""
    letters = "ABCDE"[: len(sample.choices)]
    gold_letter = sample.answers[0].strip().upper()[:1]
    if gold_letter not in letters:
        raise ValueError(f"{sample.sample_id}: gold {gold_letter!r} not among {letters}")
    gold_idx = letters.index(gold_letter)
    order = list(range(len(sample.choices)))
    rng = random.Random(f"ppu-r2::{sample.sample_id}::{seed}")
    rng.shuffle(order)
    if order == list(range(len(order))) and len(order) > 1:
        order = order[::-1]  # a "permutation" that changes nothing is not a control
    new_choices = tuple(sample.choices[i] for i in order)
    return new_choices, letters[order.index(gold_idx)]


# Meaning-preserving rewrites, applied in order, first match wins. Kept small
# and mechanical on purpose: a paraphrase that changes difficulty would show up
# as a fake instability in the label-reliability numbers.
_REWRITES: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"^what is (.+)\?$", re.I), r"Looking at the image, what is \1?"),
    (re.compile(r"^what are (.+)\?$", re.I), r"Looking at the image, what are \1?"),
    (re.compile(r"^what (.+)\?$", re.I), r"Based on the image, what \1?"),
    (re.compile(r"^who (.+)\?$", re.I), r"Based on the image, who \1?"),
    (re.compile(r"^where (.+)\?$", re.I), r"Based on the image, where \1?"),
    (re.compile(r"^when (.+)\?$", re.I), r"Based on the image, when \1?"),
    (re.compile(r"^which (.+)\?$", re.I), r"Based on the image, which \1?"),
    (re.compile(r"^how (.+)\?$", re.I), r"Based on the image, how \1?"),
    (re.compile(r"^is (.+)\?$", re.I), r"Based on the image, is \1?"),
    (re.compile(r"^are (.+)\?$", re.I), r"Based on the image, are \1?"),
    (re.compile(r"^does (.+)\?$", re.I), r"Based on the image, does \1?"),
    (re.compile(r"^do (.+)\?$", re.I), r"Based on the image, do \1?"),
)


def paraphrase(question: str) -> str:
    q = question.strip()
    for pat, repl in _REWRITES:
        if pat.match(q):
            return pat.sub(repl, q)
    return f"Based on the image: {q}" if q else q


def serve(sample: Sample, replay: int, seed: int = 7) -> ServedQuestion:
    if replay not in (1, 2):
        raise ValueError(f"replay must be 1 or 2, got {replay}")
    if sample.metric == "mcq":
        if replay == 1:
            return ServedQuestion(sample.question, sample.choices,
                                  sample.answers[0].strip().upper()[:1])
        choices, gold = _permute_choices(sample, seed)
        return ServedQuestion(sample.question, choices, gold)
    text = sample.question if replay == 1 else paraphrase(sample.question)
    return ServedQuestion(text, (), "")


def scoring_sample(sample: Sample, served: ServedQuestion) -> Sample:
    """The sample as it must be scored under this replay.

    For permuted MCQ the correct letter moves, so scoring against the original
    gold letter would silently mark every R2 rollout wrong.
    """
    if sample.metric != "mcq":
        return sample
    return Sample(
        sample_id=sample.sample_id, dataset=sample.dataset,
        image_path=sample.image_path, question=sample.question,
        answers=(served.gold_letter,), metric="mcq", choices=served.choices,
        native_hw=sample.native_hw, native_tokens=sample.native_tokens,
        meta=sample.meta,
    )
