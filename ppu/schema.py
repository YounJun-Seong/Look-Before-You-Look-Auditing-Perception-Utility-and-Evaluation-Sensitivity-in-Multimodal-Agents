"""Frozen record schema for PPU perception rollouts.

Frozen at design time (2026-08-16) BEFORE the runner exists, per SPEC_GATE01 §11:
adding a field later means re-running every rollout. Anything Gate 1's audit
table needs (§7 of the spec) must be derivable from these records alone —
including the fields that only matter for the controls (replay, dummy,
native_tokens), which are therefore present from the first logged run.

Records are JSONL, one object per line, discriminated by "type":
    {"type": "run_meta", ...}   once per runner invocation
    {"type": "rollout", ...}    once per (sample, replay, action)

Raw generations are never discarded: scoring rules will change, and every
score in this project must be recomputable from the log without GPU time.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = "0.3.0"  # 0.2.0 realized views; 0.3.0 answer-token confidence

# Closed vocabularies. Extending one is a schema version bump.
ACTION_TYPES = ("STOP", "CROP", "OCR", "HIGHRES")
DUMMY_TYPES = ("NOOP", "CROP_RANDOM", "REENCODE")
# FULL is never a candidate action: it is the full-information reference row
# that separates our question ("how to spend a fixed perceptual budget") from
# the one 2606.02357 answered ("do tools add capability at full resolution").
REFERENCE_TYPES = ("FULL",)
ALL_ACTION_TYPES = ACTION_TYPES + DUMMY_TYPES + REFERENCE_TYPES
REGION_SOURCES = ("grid", "vlm_proposal", "random", "none")
REPLAYS = (1, 2)  # R1 = canonical, R2 = permuted choices / paraphrased question

# Cost model (SPEC §3.4). Visual-token deltas on top of the coarse observation,
# which every arm has already paid for.
COARSE_TOKENS = 256
HIGHRES_TOKENS = 1280
CROP_TOKENS = 256

# Cost order used for tie-breaking "which action is best" (SPEC §7.1).
# Cheapest wins ties, otherwise the best-action distribution is decided by
# whatever order argmax happens to scan in.
COST_RANK = {"STOP": 0, "OCR": 1, "CROP": 2, "HIGHRES": 3,
             "NOOP": 0, "REENCODE": 1, "CROP_RANDOM": 2, "FULL": 4}
FULL_TOKENS = 2560  # reference budget; the log records the tokens actually used


@dataclass(frozen=True)
class Sample:
    """One dataset item, normalized across TextVQA / DocVQA / GQA / MME-RealWorld."""

    sample_id: str
    dataset: str
    image_path: str
    question: str
    answers: tuple[str, ...]  # gold answers (VQA: 10 annotations; MCQ: single letter)
    metric: str  # "vqa_acc" | "anls" | "exact" | "mcq"
    choices: tuple[str, ...] = ()  # MCQ only, in canonical order
    native_hw: tuple[int, int] = (0, 0)
    native_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metric not in ("vqa_acc", "anls", "exact", "mcq"):
            raise ValueError(f"unknown metric {self.metric!r}")
        if self.metric == "mcq" and not self.choices:
            raise ValueError(f"{self.sample_id}: mcq sample without choices")


@dataclass(frozen=True)
class ActionSpec:
    """A candidate perception action. Fully determined before execution.

    `box` is normalized (x0, y0, x1, y1) in [0, 1] and is the ONLY spatial
    information a future predictor is allowed to see for a CROP before paying
    for it (SPEC §20 of the plan / §7 of the spec).
    """

    type: str
    slot: int = 0  # CROP candidate index (0-7); 0 for non-crop actions
    box: Optional[tuple[float, float, float, float]] = None
    source: str = "none"  # REGION_SOURCES

    def __post_init__(self) -> None:
        if self.type not in ALL_ACTION_TYPES:
            raise ValueError(f"unknown action type {self.type!r}")
        if self.type in ("CROP", "CROP_RANDOM") and self.box is None:
            raise ValueError(f"{self.type} requires a box")

    @property
    def key(self) -> str:
        """Stable identifier used for caching and for joins in analysis."""
        if self.box is None:
            return f"{self.type}#{self.slot}"
        b = ",".join(f"{v:.4f}" for v in self.box)
        return f"{self.type}#{self.slot}#{b}"

    @property
    def family(self) -> str:
        """Action *type* used for diversity / exclusive-solve accounting."""
        return {"CROP_RANDOM": "CROP", "NOOP": "STOP", "REENCODE": "HIGHRES"}.get(
            self.type, self.type
        )

    @property
    def is_dummy(self) -> bool:
        return self.type in DUMMY_TYPES


@dataclass
class Cost:
    """What an action actually spent. Recorded even though Gate 1 does not
    price it into the utility (SPEC §3.4): Phase 2 needs the measurements."""

    visual_tokens: int = 0
    text_tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    tool_calls: int = 0
    ocr_latency_s: float = 0.0


@dataclass
class RolloutRecord:
    """One (sample, replay, action) execution. SPEC §11."""

    run_id: str
    replay: int
    dummy: bool
    sample_id: str
    dataset: str
    image_path: str
    native_hw: tuple[int, int]
    native_tokens: int
    question: str
    question_served: str  # R2 paraphrase / permuted-choice rendering actually sent
    gold: tuple[str, ...]
    state_id: str  # "s0" for the one-step Gate 1; kept for the sequential phase
    history: tuple[str, ...]
    action: dict[str, Any]
    obs_summary: str
    views: list[dict[str, Any]]  # realized observations: label, tokens, size, sha
    answer_before: str
    answer_after: str
    score_before: float
    score_after: float
    parse_ok: bool
    cost: dict[str, Any]
    model_id: str
    prompt_version: str
    decode: dict[str, Any]
    config_hash: str
    raw_generation: str
    confidence: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    schema_version: str = SCHEMA_VERSION
    type: str = "rollout"

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


def config_hash(payload: dict[str, Any]) -> str:
    """Stable hash of everything that could change a generation.

    Any change here invalidates cache entries, which is the point: a silent
    prompt edit that reuses cached answers would corrupt the whole audit.
    """
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]
