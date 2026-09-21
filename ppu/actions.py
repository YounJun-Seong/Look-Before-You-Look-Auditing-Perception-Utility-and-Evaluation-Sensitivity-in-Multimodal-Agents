"""The four perception actions behind one interface.

Every action — STOP included — takes the current state and returns an
observation plus the cost of having produced it. STOP is a real candidate, not
a baseline: in the audit it competes on the same footing as the others, which
is the only way "not looking" can ever be measured as the right choice.

    STOP      coarse view only, no new information, no new cost
    CROP      coarse view + region re-encoded from the ORIGINAL image (+256 tok)
    OCR       coarse view + text read from the ORIGINAL image by an external tool
    HIGHRES   the same scene re-encoded at 1280 tokens, replacing coarse

CROP keeps the coarse view alongside the zoom (context is not thrown away),
while HIGHRES replaces it (it is the same scene at a higher budget, so keeping
both would double-charge for the same content).

Dummy actions (NOOP / CROP_RANDOM / REENCODE) share this interface so the
luck-floor arm runs through exactly the same code path as the real one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import imaging
from .imaging import View
from .ocr import OCREngine, OCRResult
from .schema import (COARSE_TOKENS, CROP_TOKENS, FULL_TOKENS, HIGHRES_TOKENS,
                     ActionSpec, Cost, Sample)


@dataclass
class Observation:
    views: list[View]
    aux_text: str
    cost: Cost
    summary: str
    ocr: Optional[OCRResult] = None


@dataclass
class ActionExecutor:
    coarse_tokens: int = COARSE_TOKENS
    highres_tokens: int = HIGHRES_TOKENS
    crop_tokens: int = CROP_TOKENS
    full_tokens: int = FULL_TOKENS
    ocr_engine: Optional[OCREngine] = None
    _coarse_memo: dict = field(default_factory=dict)

    def coarse(self, sample: Sample) -> View:
        if sample.image_path not in self._coarse_memo:
            self._coarse_memo[sample.image_path] = imaging.coarse_view(
                sample.image_path, self.coarse_tokens
            )
        return self._coarse_memo[sample.image_path]

    def observe(self, sample: Sample, spec: ActionSpec) -> Observation:
        base = self.coarse(sample)
        t = spec.type

        if t == "STOP":
            return Observation([base], "", Cost(visual_tokens=0), "stop")

        if t == "NOOP":
            # Same observation shown twice: pure "ask again" control.
            return Observation([base, base], "", Cost(visual_tokens=base.tokens),
                               "noop-repeat")

        if t == "REENCODE":
            v = imaging.reencode_view(sample.image_path, self.coarse_tokens, spec.slot)
            return Observation([v], "", Cost(visual_tokens=0), f"reencode{spec.slot}")

        if t in ("CROP", "CROP_RANDOM"):
            crop = imaging.crop_view(sample.image_path, spec.box, self.crop_tokens)
            return Observation([base, crop], "", Cost(visual_tokens=crop.tokens),
                               f"crop@{crop.tokens}tok {spec.box}")

        if t == "HIGHRES":
            v = imaging.highres_view(sample.image_path, self.highres_tokens)
            return Observation([v], "", Cost(visual_tokens=v.tokens - base.tokens),
                               f"highres@{v.tokens}tok")

        if t == "FULL":
            # full_tokens <= 0 means uncapped: FULL is the native view, i.e. the
            # information ceiling. The cap exists only to keep gigapixel images
            # servable (MME-RealWorld); TextVQA needs no cap.
            cap = self.full_tokens if self.full_tokens > 0 else 10 ** 9
            v = imaging.full_view(sample.image_path, cap)
            return Observation([v], "", Cost(visual_tokens=v.tokens - base.tokens),
                               f"full@{v.tokens}tok")

        if t == "OCR":
            if self.ocr_engine is None:
                raise RuntimeError("OCR action requested but no OCR engine configured")
            res = self.ocr_engine.read(sample.image_path)
            return Observation(
                views=[base],
                aux_text=res.serialize(),
                cost=Cost(visual_tokens=0, tool_calls=1, ocr_latency_s=res.latency_s),
                summary=f"ocr:{len(res.lines)}lines",
                ocr=res,
            )

        raise ValueError(f"unhandled action type {t!r}")


def view_labels(obs: Observation, spec: ActionSpec) -> list[str]:
    """Human-readable description of each view, in order.

    Says *where* a crop came from but never what is in it — the coordinates are
    part of the action spec, which a predictor is allowed to see before paying;
    the pixels are not.
    """
    labels = []
    for v in obs.views:
        if v.label.startswith("crop") and v.box:
            x0, y0, x1, y1 = v.box
            labels.append(
                f"zoomed view of region ({x0:.2f}, {y0:.2f}) to ({x1:.2f}, {y1:.2f})"
            )
        elif v.label.startswith(("highres", "full")):
            labels.append("higher-resolution view of the whole image")
        else:
            labels.append("low-resolution overview of the whole image")
    return labels
