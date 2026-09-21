#!/usr/bin/env python3
"""CSV variant of the adjudication package (same 300 items, same blinding):
paper/human_adjudication_package/
  annotator1.csv, annotator2.csv, annotator3.csv   fill the `verdict` column with
                                                  C / I / A (and optional `note`)
  images/<item_id>.jpg                            the image for each row (max 1024 px)
  README.txt                                      instruction (verbatim from the prereg)
and paper/human_adjudication_package.zip. Rows are in the same shuffled order
as the HTML sheet; no scorer / arm / policy information is included.
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/human_adjudication_package"
INSTR = """Human semantic adjudication — instructions

Judge whether the model prediction correctly answers the question in context.
Mark C (Correct) only when its overall meaning is consistent with the reference
answers and does not introduce a contradiction or an incompatible alternative.
A prediction that merely contains the reference token somewhere is not
necessarily correct. Use A (Ambiguous) when the available image / question /
references do not support a confident judgment. Otherwise mark I (Incorrect).
Surface form (casing, punctuation, extra function words) does not matter;
meaning does.

How to fill: open your annotatorN.csv, look at images/<item_id>.jpg for each
row, and type C, I or A in the `verdict` column (optional free text in `note`).
Save as CSV (UTF-8). Do not discuss items with the other annotators before
everyone has finished. Reference answers are listed with their vote counts
(e.g. "yes x8; no x2").
"""


def main():
    items = [json.loads(l) for l in open(ROOT / "data/human_adjudication/items.jsonl")]
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "README.txt").write_text(INSTR)
    rows = []
    for it in items:
        im = Image.open(ROOT / it["image_path"]).convert("RGB"); im.thumbnail((800, 800))
        im.save(OUT / "images" / f"{it['item_id']}.jpg", quality=75)
        c = Counter(x.strip().lower() for x in it["gold"])
        rows.append({"item_id": it["item_id"], "image": f"images/{it['item_id']}.jpg",
                     "question": it["question"],
                     "reference_answers": "; ".join(f"{a} x{n}" for a, n in c.most_common()),
                     "model_prediction": it["answer_raw"].replace("\n", " "),
                     "verdict": "", "note": ""})
    for k in (1, 2, 3):
        with open(OUT / f"annotator{k}.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), quoting=csv.QUOTE_ALL)
            w.writeheader(); w.writerows(rows)
    zip_path = shutil.make_archive(str(ROOT / "paper/human_adjudication_package"), "zip", OUT.parent, OUT.name)
    print(f"{len(rows)} rows × 3 CSVs + {len(rows)} images -> {zip_path} ({Path(zip_path).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
