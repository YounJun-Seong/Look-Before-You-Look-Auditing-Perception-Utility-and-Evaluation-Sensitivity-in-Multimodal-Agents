"""CROP candidate generation (SPEC §3.3).

Exactly K=8 candidates per sample, always. Constant cardinality matters more
than candidate quality here: the oracle takes a max over the candidate set, so
a sample with 11 candidates gets a bigger selection-on-noise bonus than one
with 6, and the "which action type wins" distribution would drift with it.

Composition: 5 multi-scale grid boxes (detector-independent, so the result is
not an artifact of one proposal model) + 3 VLM-proposed boxes. Object/text
proposal networks are deliberately excluded from Gate 1 — their cost and their
failure modes would land inside our numbers.

The proposal call is one VLM generation and is charged to every arm that
considers CROP at all (SPEC §3.4), including Random and Oracle.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Optional

from .schema import ActionSpec

Box = tuple[float, float, float, float]
K_CANDIDATES = 8
MIN_AREA = 0.01  # 1% of the image; below this a crop is mostly upsampled noise

GRID_BOXES: tuple[Box, ...] = (
    (0.25, 0.25, 0.75, 0.75),  # center
    (0.00, 0.00, 0.55, 0.55),  # top-left
    (0.45, 0.00, 1.00, 0.55),  # top-right
    (0.00, 0.45, 0.55, 1.00),  # bottom-left
    (0.45, 0.45, 1.00, 1.00),  # bottom-right
)

# Used when the proposal call fails or returns garbage: small boxes, so the
# fallback still probes a different scale than the grid rather than duplicating it.
FALLBACK_BOXES: tuple[Box, ...] = (
    (0.35, 0.35, 0.65, 0.65),
    (0.05, 0.05, 0.35, 0.35),
    (0.65, 0.65, 0.95, 0.95),
)

_NUM = re.compile(r"[-+]?\d*\.?\d+")


def clip_box(box: Box) -> Optional[Box]:
    x0, y0, x1, y1 = box
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    if (x1 - x0) * (y1 - y0) < MIN_AREA:
        return None
    return (round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4))


def iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def parse_proposals(text: str, limit: int = 3) -> list[Box]:
    """Read boxes out of a proposal generation. Tolerant, but never creative:
    a line that does not yield four numbers in [0,1] is dropped, not repaired."""
    boxes: list[Box] = []
    for line in (text or "").splitlines():
        nums = _NUM.findall(line)
        if len(nums) < 4:
            continue
        try:
            vals = [float(n) for n in nums[:4]]
        except ValueError:
            continue
        if any(v < -0.01 or v > 1.01 for v in vals):
            continue
        box = clip_box((vals[0], vals[1], vals[2], vals[3]))
        if box is not None:
            boxes.append(box)
        if len(boxes) >= limit:
            break
    return boxes


def parse_proposals_mixed(text: str, view_w: int, view_h: int,
                          limit: int = 3) -> tuple[list[Box], list[dict]]:
    """Parse proposals that mix normalized and pixel coordinates.

    Qwen2.5-VL is trained to ground in absolute pixels, and asking for
    fractions makes it emit both in the same line:

        0.25, 147, 0.38, 236   ->  x are fractions, y are pixels

    The original parser required all four values in [0, 1] and dropped 70% of
    the calls, so the question-conditioned arm of the audit was almost never
    executed. Each coordinate is therefore converted independently against the
    dimension it belongs to (the view the model actually saw), and every box
    carries how it was read so the decision is auditable later.
    """
    boxes: list[Box] = []
    trace: list[dict] = []
    for line in (text or "").splitlines():
        nums = _NUM.findall(line)
        if len(nums) < 4:
            continue
        try:
            raw = [float(n) for n in nums[:4]]
        except ValueError:
            continue
        dims = (view_w, view_h, view_w, view_h)
        modes = ["norm" if abs(v) <= 1.0 else "pixel" for v in raw]
        vals = [v if m == "norm" else v / d
                for v, m, d in zip(raw, modes, dims)]
        rec = {"raw": raw, "modes": modes,
               "normalized": [round(v, 4) for v in vals]}
        if any(v < -0.01 or v > 1.01 for v in vals):
            rec["dropped"] = "out_of_range"
            trace.append(rec)
            continue
        box = clip_box((vals[0], vals[1], vals[2], vals[3]))
        if box is None:
            rec["dropped"] = "degenerate_or_too_small"
            trace.append(rec)
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > 0.95:
            rec["dropped"] = "near_full_image"
            trace.append(rec)
            continue
        if any(iou(box, b) > 0.9 for b in boxes):
            rec["dropped"] = "duplicate"
            trace.append(rec)
            continue
        rec["box"] = box
        rec["area"] = round(area, 4)
        trace.append(rec)
        boxes.append(box)
        if len(boxes) >= limit:
            break
    return boxes, trace


def size_matched_random(sample_id: str, boxes: list[Box],
                        seed_tag: str = "matched") -> list[Box]:
    """Random boxes with the SAME area and aspect ratio, only the position moves.

    Without this, "question-conditioned beats random" could just mean the
    proposals were smaller (or larger) than the random ones — a crop-size
    effect wearing the costume of a selection effect.
    """
    rng = random.Random(f"ppu-{seed_tag}::{sample_id}")
    out: list[Box] = []
    for b in boxes:
        w, h = b[2] - b[0], b[3] - b[1]
        x0 = rng.uniform(0.0, max(0.0, 1.0 - w))
        y0 = rng.uniform(0.0, max(0.0, 1.0 - h))
        nb = clip_box((x0, y0, x0 + w, y0 + h))
        out.append(nb if nb is not None else b)
    return out


def _dedupe(boxes: list[Box], thresh: float = 0.9) -> list[Box]:
    kept: list[Box] = []
    for b in boxes:
        if all(iou(b, k) <= thresh for k in kept):
            kept.append(b)
    return kept


def build_candidates(proposals: list[Box]) -> list[ActionSpec]:
    """Grid ∪ proposals, deduped, padded to exactly K_CANDIDATES."""
    specs: list[ActionSpec] = []
    sources: list[str] = []
    boxes: list[Box] = []

    for b in GRID_BOXES:
        boxes.append(b)
        sources.append("grid")
    for b in proposals:
        boxes.append(b)
        sources.append("vlm_proposal")

    kept_boxes: list[Box] = []
    kept_sources: list[str] = []
    for b, s in zip(boxes, sources):
        cb = clip_box(b)
        if cb is None:
            continue
        if all(iou(cb, k) <= 0.9 for k in kept_boxes):
            kept_boxes.append(cb)
            kept_sources.append(s)

    for b in FALLBACK_BOXES:
        if len(kept_boxes) >= K_CANDIDATES:
            break
        cb = clip_box(b)
        if cb is not None and all(iou(cb, k) <= 0.9 for k in kept_boxes):
            kept_boxes.append(cb)
            kept_sources.append("grid")

    # Still short (heavy overlap): shrink the center box in fixed steps. Fully
    # deterministic so the candidate set is reproducible from the sample alone.
    step = 0
    while len(kept_boxes) < K_CANDIDATES:
        scale = 0.45 - 0.05 * step
        step += 1
        if scale <= 0.12:
            break
        half = scale / 2
        cb = clip_box((0.5 - half, 0.5 - half, 0.5 + half, 0.5 + half))
        if cb is not None and all(iou(cb, k) <= 0.9 for k in kept_boxes):
            kept_boxes.append(cb)
            kept_sources.append("grid")

    for i, (b, s) in enumerate(zip(kept_boxes[:K_CANDIDATES], kept_sources[:K_CANDIDATES])):
        specs.append(ActionSpec(type="CROP", slot=i, box=b, source=s))
    return specs


def random_boxes(sample_id: str, reference: list[ActionSpec], n: int = 8) -> list[ActionSpec]:
    """Scale-matched random boxes for the dummy arm (SPEC §6.3).

    Same number, same size distribution, question-independent placement: the
    difference between the real oracle and the dummy oracle is then attributable
    to *where* we looked, not to how many times we asked.
    """
    rng = random.Random(f"ppu-dummy::{sample_id}")
    sizes = [(b.box[2] - b.box[0], b.box[3] - b.box[1]) for b in reference if b.box]
    if not sizes:
        sizes = [(0.5, 0.5)] * n
    out: list[ActionSpec] = []
    for i in range(n):
        w, h = sizes[i % len(sizes)]
        x0 = rng.uniform(0.0, max(0.0, 1.0 - w))
        y0 = rng.uniform(0.0, max(0.0, 1.0 - h))
        box = clip_box((x0, y0, x0 + w, y0 + h))
        if box is None:
            box = (0.25, 0.25, 0.75, 0.75)
        out.append(ActionSpec(type="CROP_RANDOM", slot=i, box=box, source="random"))
    return out
