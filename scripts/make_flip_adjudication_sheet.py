#!/usr/bin/env python3
"""Semantic-adjudication sheet for the VQAv2 lockbox format-flip HELP states.

The strict scorer says HIGHRES fixed these 28 states; the format-confound
analysis says most of that is a multi-word STOP reply being re-emitted as one
word. The direct test of "measured PPU != evidence value" is a human reading
each STOP answer against the golds: if the STOP answer was already
semantically correct, the strict HELP label rewards formatting, not evidence.

Selection (matches scripts/format_confound_check.py): strict cross-replay
HELP, STOP R1 answer multi-word, HIGHRES R1 answer single-word, on
logs/lockbox_vqav2.jsonl.

Outputs:
  data/adjudication_flip28.jsonl      machine-readable rows
  paper/adjudication_flip28.md        annotation sheet (fill `verdict`)
  paper/adjudication_flip28.html      same rows with embedded coarse images

Verdict codes to use: CORRECT (STOP answer semantically matches the gold
consensus), PARTIAL (right object/property, wrong granularity or hedged),
WRONG (semantically incorrect). Annotate from the question + golds; the image
is shown at the coarse budget the model actually saw.
"""
from __future__ import annotations

import base64
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LOG = "logs/lockbox_vqav2.jsonl"
THR = 0.5


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0:
        return "HELP"
    if d1 < 0 and d2 < 0:
        return "HARM"
    if d1 == 0 and d2 == 0:
        return "NEUTRAL"
    return "AMBIGUOUS"


def gold_summary(golds):
    return ", ".join(f"{a} x{c}" for a, c in Counter(g.lower() for g in golds).most_common())


def thumb_b64(path, width=360):
    img = Image.open(ROOT / path).convert("RGB")
    h = int(img.height * width / img.width)
    img = img.resize((width, h))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    by = defaultdict(dict)
    for line in open(ROOT / LOG):
        r = json.loads(line)
        if r.get("type") == "rollout":
            by[r["sample_id"]][(r["action"]["type"], r["replay"])] = r

    rows = []
    for sid, d in sorted(by.items()):
        need = [("STOP", 1), ("STOP", 2), ("HIGHRES", 1), ("HIGHRES", 2)]
        if not all(k in d for k in need):
            continue
        sc = {k: float(d[k]["score_after"]) for k in need}
        lab = four_way(sc[("STOP", 1)], sc[("STOP", 2)], sc[("HIGHRES", 1)], sc[("HIGHRES", 2)])
        st, hi = d[("STOP", 1)], d[("HIGHRES", 1)]
        sa, ha = (st["answer_after"] or ""), (hi["answer_after"] or "")
        if lab == "HELP" and len(sa.split()) > 1 and len(ha.split()) <= 1:
            rows.append({
                "sample_id": sid, "image_path": st["image_path"],
                "question": st["question"], "gold": st["gold"],
                "stop_answer_r1": sa, "stop_answer_r2": d[("STOP", 2)]["answer_after"],
                "highres_answer_r1": ha, "highres_answer_r2": d[("HIGHRES", 2)]["answer_after"],
                "stop_strict": [sc[("STOP", 1)], sc[("STOP", 2)]],
                "highres_strict": [sc[("HIGHRES", 1)], sc[("HIGHRES", 2)]],
                "verdict": "", "note": "",
            })
    print(f"flip HELP states: {len(rows)}")

    with open(ROOT / "data/adjudication_flip28.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    md = ["# VQAv2 lockbox — format-flip HELP states: semantic adjudication",
          "",
          f"n = {len(rows)}. Strict HELP where STOP (R1) is multi-word and HIGHRES (R1)",
          "is one word. Fill `verdict` with CORRECT / PARTIAL / WRONG for the STOP",
          "answer against the gold consensus. Post-hoc; annotate before looking at",
          "any aggregate result.",
          "",
          "| # | question | golds | STOP answer (R1 / R2) | HIGHRES answer (R1 / R2) | verdict | note |",
          "|--:|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        md.append(f"| {i} | {r['question']} | {gold_summary(r['gold'])} | "
                  f"{r['stop_answer_r1']} / {r['stop_answer_r2']} | "
                  f"{r['highres_answer_r1']} / {r['highres_answer_r2']} |  |  |")
    (ROOT / "paper/adjudication_flip28.md").write_text("\n".join(md) + "\n")

    html = ["<meta charset='utf-8'><style>body{font-family:sans-serif;max-width:900px;"
            "margin:2em auto}div.c{border:1px solid #ccc;border-radius:6px;padding:1em;"
            "margin:1em 0}img{border:1px solid #999}code{background:#f2f2f2;padding:1px 4px}"
            "</style><h2>VQAv2 lockbox — format-flip HELP states (n=%d)</h2>"
            "<p>Image shown near the coarse budget the model saw. Verdict the STOP "
            "answer: CORRECT / PARTIAL / WRONG.</p>" % len(rows)]
    for i, r in enumerate(rows, 1):
        html.append(
            f"<div class='c'><b>#{i}</b> <code>{r['sample_id']}</code><br>"
            f"<img src='data:image/jpeg;base64,{thumb_b64(r['image_path'])}'><br>"
            f"<b>Q:</b> {r['question']}<br>"
            f"<b>golds:</b> {gold_summary(r['gold'])}<br>"
            f"<b>STOP:</b> {r['stop_answer_r1']}  /  {r['stop_answer_r2']}<br>"
            f"<b>HIGHRES:</b> {r['highres_answer_r1']}  /  {r['highres_answer_r2']}<br>"
            f"<b>verdict:</b> ______</div>")
    (ROOT / "paper/adjudication_flip28.html").write_text("\n".join(html))
    print("wrote data/adjudication_flip28.jsonl, paper/adjudication_flip28.{md,html}")


if __name__ == "__main__":
    main()
