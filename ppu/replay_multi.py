"""Replays 3-5 for the multi-replay robustness experiment.

The frozen two-replay module (ppu/replay.py) is untouched: replays 1 and 2
delegate to it verbatim, so every existing log remains reproducible. Replays
3-5 extend the same two surface axes it already uses:

  * MCQ: a fresh deterministic choice permutation per replay (seeds fixed
    here, gold letter tracked exactly as replay 2 does);
  * open-ended: a fixed semantics-preserving wrapper of the question, in the
    same colon/suffix style as the frozen fallback ("Based on the image: ...").

Everything is derivable from the manifest + this file alone. Prereg:
paper/PREREG_MULTI_REPLAY.md.
"""
from __future__ import annotations

from . import replay as _frozen
from .replay import ServedQuestion, _permute_choices, scoring_sample  # noqa: F401
from .schema import Sample

MCQ_SEEDS = {3: 13, 4: 29, 5: 43}
OPEN_WRAP = {
    3: lambda q: f"In the image shown: {q}",
    4: lambda q: f"{q} Answer from the image.",
    5: lambda q: f"Consider the picture: {q}",
}
MAX_REPLAY = 5


def serve(sample: Sample, replay: int, seed: int = 7) -> ServedQuestion:
    if replay in (1, 2):
        return _frozen.serve(sample, replay, seed)
    if replay not in MCQ_SEEDS:
        raise ValueError(f"replay must be 1..{MAX_REPLAY}, got {replay}")
    if sample.metric == "mcq":
        choices, gold = _permute_choices(sample, MCQ_SEEDS[replay])
        return ServedQuestion(sample.question, choices, gold)
    return ServedQuestion(OPEN_WRAP[replay](sample.question.strip()), (), "")
