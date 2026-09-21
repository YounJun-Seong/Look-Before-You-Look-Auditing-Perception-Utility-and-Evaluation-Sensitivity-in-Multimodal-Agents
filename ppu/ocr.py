"""OCR as an external perception tool.

Two design decisions, both from SPEC §3.5:

1. The OCR engine is NOT the VLM. If we asked the reader to transcribe the
   image at high resolution, OCR and HIGHRES would be the same action and the
   action-diversity result would be an artifact of prompt wording.
2. OCR reads the ORIGINAL image, not the coarse view — reading text the reader
   cannot resolve is the entire reason the tool exists.

OCR is not free: engine latency and the tool call are recorded in Cost, and
the returned text is an *observation*, produced only after the action is
chosen. Nothing upstream of the choice may read it (SPEC §7 / plan §20).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

MAX_LINES = 30


@dataclass
class OCRLine:
    text: str
    box: tuple[float, float, float, float]  # normalized x0,y0,x1,y1
    conf: float = 1.0


@dataclass
class OCRResult:
    lines: list[OCRLine]
    latency_s: float = 0.0
    engine: str = ""

    def serialize(self, max_lines: int = MAX_LINES) -> str:
        """Reading-order text block handed to the reader."""
        kept = sorted(self.lines, key=lambda l: (round(l.box[1], 2), l.box[0]))[:max_lines]
        if not kept:
            return "(no text detected)"
        return "\n".join(
            f"[{l.box[0]:.2f},{l.box[1]:.2f},{l.box[2]:.2f},{l.box[3]:.2f}] {l.text}"
            for l in kept
        )


class OCREngine(Protocol):
    name: str

    def read(self, image_path: str) -> OCRResult: ...


@dataclass
class RapidOCREngine:
    """RapidOCR (ONNX, CPU). Loaded lazily so the light venv stays importable
    on machines without the model files."""

    name: str = "rapidocr"
    _engine: object = None

    def _ensure(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    def read(self, image_path: str) -> OCRResult:
        from PIL import Image

        engine = self._ensure()
        t0 = time.time()
        result, _ = engine(image_path)
        latency = time.time() - t0
        with Image.open(image_path) as im:
            w, h = im.size
        lines: list[OCRLine] = []
        for item in result or []:
            poly, text, conf = item[0], item[1], float(item[2])
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            lines.append(OCRLine(
                text=text,
                box=(min(xs) / w, min(ys) / h, max(xs) / w, max(ys) / h),
                conf=conf,
            ))
        return OCRResult(lines=lines, latency_s=latency, engine=self.name)


@dataclass
class StubOCR:
    """Offline engine for tests: returns whatever the fixture registered."""

    name: str = "stub-ocr"
    table: dict = field(default_factory=dict)  # image_path -> list[OCRLine]

    def read(self, image_path: str) -> OCRResult:
        return OCRResult(lines=list(self.table.get(image_path, [])),
                         latency_s=0.001, engine=self.name)


@dataclass
class PrecomputedOCR:
    """Serves OCR from a pre-pass file (scripts/ocr_prepass.py).

    Reading a 5000x5000 remote-sensing image costs ~20-30s single-threaded and
    would dominate the campaign wall clock while the GPU idles. The pre-pass
    does every image once across CPU workers; the campaign then reads results.
    The recorded latency is the engine's real cost, not the lookup's, so the
    cost column still reports what a deployed agent would pay.
    """

    path: str
    fallback: Optional[OCREngine] = None
    name: str = "rapidocr-prepass"
    _table: Optional[dict] = None
    hits: int = 0
    misses: int = 0

    def _load(self) -> dict:
        if self._table is None:
            import json
            from pathlib import Path

            raw = json.loads(Path(self.path).read_text()) if Path(self.path).exists() else {}
            self._table = raw
        return self._table

    def read(self, image_path: str) -> OCRResult:
        table = self._load()
        entry = table.get(image_path)
        if entry is None:
            self.misses += 1
            if self.fallback is None:
                raise KeyError(f"no OCR pre-pass entry for {image_path}")
            res = self.fallback.read(image_path)
            table[image_path] = {"lines": [[l.text, list(l.box), l.conf] for l in res.lines],
                                 "latency_s": res.latency_s}
            return res
        self.hits += 1
        return OCRResult(
            lines=[OCRLine(t, tuple(b), c) for t, b, c in entry["lines"]],
            latency_s=float(entry.get("latency_s", 0.0)),
            engine=self.name,
        )


@dataclass
class CachedOCR:
    """Wraps an engine so each image is read at most once per campaign.

    The cached result keeps the engine's original latency, so every OCR rollout
    is charged what the tool actually costs to run. The campaign saves the
    repeat reads; a deployed agent would not, and the cost column reports the
    deployed price, not our bookkeeping shortcut.
    """

    engine: OCREngine
    _memo: dict = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    @property
    def name(self) -> str:
        return getattr(self.engine, "name", "unknown")

    def read(self, image_path: str) -> OCRResult:
        if image_path not in self._memo:
            self.misses += 1
            self._memo[image_path] = self.engine.read(image_path)
        else:
            self.hits += 1
        return self._memo[image_path]
