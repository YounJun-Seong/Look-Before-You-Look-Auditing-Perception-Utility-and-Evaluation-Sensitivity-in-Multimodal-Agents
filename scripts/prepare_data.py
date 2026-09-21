#!/usr/bin/env python
"""Build sample manifests, once, before any rollout runs.

Each source becomes a JSONL of `Sample` records under data/manifests/. The
runner never parses a dataset format, so the exact sample set is a committed
artifact rather than whatever the loader did on the day.

Native token counts are computed here (they gate inclusion, SPEC §3.1), which
means this script opens every image — slow once, free afterwards.

Sources (expected on disk, downloaded separately):
    textvqa   : TextVQA_0.5.1_val.json  + train_images/
    docvqa    : val/val_v1.0_withQT.json + val/documents/
    gqa       : val_balanced_questions.json + images/
    mme_realworld : MME_RealWorld.json + images/
    synthetic : generated locally, for smoke tests without any download

Usage:
    python scripts/prepare_data.py synthetic --out data/manifests/synthetic.jsonl -n 12
    python scripts/prepare_data.py textvqa --root /data/textvqa --out data/manifests/textvqa.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

from ppu.datasets import METRIC_BY_DATASET, write_manifest
from ppu.imaging import native_tokens
from ppu.schema import Sample


def _finish(sample_id, dataset, image_path, question, answers, choices=(), meta=None):
    try:
        with Image.open(image_path) as im:
            w, h = im.size
        ntok = native_tokens(image_path)
    except Exception as exc:  # noqa: BLE001 - a broken image is a skipped sample
        print(f"  skip {sample_id}: {exc}", file=sys.stderr)
        return None
    return Sample(
        sample_id=sample_id, dataset=dataset, image_path=str(image_path),
        question=question, answers=tuple(answers), metric=METRIC_BY_DATASET[dataset],
        choices=tuple(choices), native_hw=(h, w), native_tokens=ntok, meta=meta or {},
    )


def build_synthetic(root: Path, n: int) -> list[Sample]:
    """Locally drawn images with a small legible detail and a large obvious one.

    Not a benchmark — a way to exercise the whole path (crop vs OCR vs highres)
    end to end without waiting on a download.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260816)
    out = []
    colors = ["red", "blue", "green", "orange"]
    for i in range(n):
        w, h = 2000, 1400
        im = Image.new("RGB", (w, h), "white")
        d = ImageDraw.Draw(im)
        colour = rng.choice(colors)
        d.rectangle([120, 120, 520, 420], fill=colour)
        price = f"{rng.randint(1, 99)}.{rng.randint(10, 99)}"
        d.text((int(w * 0.62), int(h * 0.78)), f"PRICE ${price}", fill="black")
        p = root / f"syn_{i:04d}.png"
        im.save(p)
        ask_price = i % 2 == 0
        s = _finish(f"synthetic/{i:04d}", "synthetic", p,
                    "What is the price on the tag?" if ask_price
                    else "What color is the large rectangle?",
                    [price] if ask_price else [colour])
        if s:
            out.append(s)
    return out


def build_textvqa(root: Path, split: str = "val") -> list[Sample]:
    ann = json.loads((root / f"TextVQA_0.5.1_{split}.json").read_text())
    img_dir = root / f"{'train' if split != 'test' else 'test'}_images"
    out = []
    for q in ann["data"]:
        p = img_dir / f"{q['image_id']}.jpg"
        if not p.exists():
            continue
        s = _finish(f"textvqa/{split}/{q['question_id']}", "textvqa", p,
                    q["question"], q.get("answers", []),
                    meta={"image_id": q["image_id"]})
        if s:
            out.append(s)
    return out


def build_docvqa(root: Path, split: str = "val") -> list[Sample]:
    ann = json.loads((root / split / f"{split}_v1.0_withQT.json").read_text())
    out = []
    for q in ann["data"]:
        p = root / split / q["image"]
        if not p.exists():
            continue
        s = _finish(f"docvqa/{split}/{q['questionId']}", "docvqa", p,
                    q["question"], q.get("answers", []),
                    meta={"question_type": q.get("question_types", [])})
        if s:
            out.append(s)
    return out


def build_gqa(root: Path, split: str = "val_balanced") -> list[Sample]:
    ann = json.loads((root / f"{split}_questions.json").read_text())
    out = []
    for qid, q in ann.items():
        p = root / "images" / f"{q['imageId']}.jpg"
        if not p.exists():
            continue
        s = _finish(f"gqa/{split}/{qid}", "gqa", p, q["question"], [q["answer"]],
                    meta={"types": q.get("types", {})})
        if s:
            out.append(s)
    return out


_CHOICE_PREFIX = re.compile(r"^\s*[\(\[]?([A-E])[\)\].:]\s*")


def build_mme_realworld(root: Path) -> list[Sample]:
    """MME-RealWorld (full or Lite). Root holds the annotation JSON and imgs/.

    Choices arrive as "(A) In the upper right area..."; the letter is stripped
    because our own renderer re-letters them — and R2 permutes the order, so a
    baked-in letter would contradict the shown one.
    """
    candidates = ["MME-RealWorld-Lite.json", "MME_RealWorld.json"]
    ann_path = next((root / c for c in candidates if (root / c).exists()), None)
    if ann_path is None:
        raise FileNotFoundError(f"no MME annotation json under {root} ({candidates})")
    ann = json.loads(ann_path.read_text())
    img_dir = root / "imgs" if (root / "imgs").is_dir() else root
    out = []
    for q in ann:
        p = img_dir / q["Image"]
        if not p.exists():
            continue
        choices = [_CHOICE_PREFIX.sub("", c).strip() for c in q["Answer choices"]]
        s = _finish(f"mme_realworld/{q['Question_id']}", "mme_realworld", p,
                    q["Text"], [q["Ground truth"]], choices,
                    meta={"task": q.get("Task"), "subtask": q.get("Subtask"),
                          "category": q.get("Category"),
                          "source": ann_path.name})
        if s:
            out.append(s)
    return out


BUILDERS = {
    "synthetic": lambda a: build_synthetic(Path(a.root or "data/synthetic"), a.n),
    "textvqa": lambda a: build_textvqa(Path(a.root), a.split or "val"),
    "docvqa": lambda a: build_docvqa(Path(a.root), a.split or "val"),
    "gqa": lambda a: build_gqa(Path(a.root), a.split or "val_balanced"),
    "mme_realworld": lambda a: build_mme_realworld(Path(a.root)),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(BUILDERS))
    ap.add_argument("--root", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=12, help="synthetic only")
    ap.add_argument("--coarse-tokens", type=int, default=256,
                    help="budget the eligibility report is computed against "
                         "(dataset-adaptive: MME 256, TextVQA 192)")
    args = ap.parse_args()

    samples = BUILDERS[args.dataset](args)

    # Source data can ship literal duplicate rows (MME-RealWorld-Lite has 77
    # duplicated Question_ids). Identical duplicates only waste rollouts, but a
    # collision between *different* rows would silently merge two questions into
    # one SampleView, so ids are made unique here and the count is reported.
    seen: dict[str, int] = {}
    unique: list[Sample] = []
    n_dup = 0
    for s_ in samples:
        if s_.sample_id in seen:
            seen[s_.sample_id] += 1
            n_dup += 1
            s_ = Sample(**{**s_.__dict__,
                           "sample_id": f"{s_.sample_id}#dup{seen[s_.sample_id]}"})
        else:
            seen[s_.sample_id] = 0
        unique.append(s_)
    samples = unique

    n = write_manifest(args.out, samples)
    if n_dup:
        print(f"  NOTE: {n_dup} duplicate sample_ids in the source were suffixed "
              f"(#dupN) to keep ids unique")
    eligible = sum(1 for s in samples if s.native_tokens >= 4 * args.coarse_tokens)
    print(f"{args.dataset}: wrote {n} samples -> {args.out}")
    print(f"  inclusion-eligible (native >= 4x coarse={args.coarse_tokens}): {eligible} "
          f"({100 * eligible / max(n, 1):.1f}%)  [SPEC G0-7]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
