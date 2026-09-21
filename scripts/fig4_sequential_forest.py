#!/usr/bin/env python3
"""Fig — prospective sequential opportunity vs realizability (strict only).

Two contrasts per first action against the frozen state-independent
continuation (= always→CROP-TR), on the 300 fresh states, strict scorer:

  ★  sequential oracle  (cross-replay counterfactual opportunity)
  ●  frozen controller  (strict-trained heads; Center = preregistered primary)

Zero line = the frozen continuation. Values annotated with 95% CIs.
Reads paper/seq_prospective_eval.json. Writes
paper/figures/fig4_sequential_forest.{pdf,png}.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
D = json.load(open(ROOT / "paper/seq_prospective_eval.json"))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.8,
                     "axes.edgecolor": "#444444", "axes.labelcolor": "#111111",
                     "xtick.color": "#333333", "ytick.color": "#222222",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
BLUE, GOLD, SUB, GRID = "#0173B2", "#C99700", "#767676", "#ECECEA"


def orc(a1):
    r = D["oracle_gap_ci"]["strict"][a1]
    return 100 * r["oracle_minus_mixture"], [100 * c for c in r["ci95_mixture"]]


def ctl(a1):
    r = D["results"]["strict"][a1]["controller_strictheads"]
    return 100 * r["minus_mixture"], [100 * c for c in r["ci95_mix"]]


ROWS = [  # (y, group, kind, a1, tag)
    (3.35, "After Center Crop", "oracle", "center", ""),
    (2.55, "", "controller", "center", "Primary"),
    (1.15, "After OCR", "oracle", "ocr", ""),
    (0.35, "", "controller", "ocr", "Secondary"),
]

fig, ax = plt.subplots(figsize=(5.2, 2.8), dpi=300)
fig.subplots_adjust(left=0.27, right=0.98, top=0.97, bottom=0.30)

for y, group, kind, a1, tag in ROWS:
    d, ci = (orc if kind == "oracle" else ctl)(a1)
    col = GOLD if kind == "oracle" else BLUE
    mk, ms = ("*", 13) if kind == "oracle" else ("o", 7)
    ax.plot(ci, [y, y], color=col, lw=1.8, solid_capstyle="round", zorder=3)
    for c in ci:
        ax.plot([c, c], [y - 0.09, y + 0.09], color=col, lw=1.3, zorder=3)
    ax.plot(d, y, mk, ms=ms, mfc=col, mec="white", mew=0.8, zorder=4)
    ax.annotate(f"{d:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]",
                (d, y - 0.34), ha="center", fontsize=7.4, color=col)
    if tag:
        ax.annotate(tag, (ci[1] + 0.55, y), va="center", fontsize=7.4,
                    color="#333333", style="italic")

# row + group labels in the left margin
lab = {"oracle": "Sequential oracle", "controller": "Frozen controller"}
for y, group, kind, a1, tag in ROWS:
    ax.text(-0.02, y, lab[kind], transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=8.4, color="#333333")
for y, g in ((3.35 + 0.55, "After Center Crop"), (1.15 + 0.55, "After OCR")):
    ax.text(-0.30, y, g, transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=9.2, color="#111111",
            fontweight="bold")

ax.axvline(0, color="#222222", lw=0.9, zorder=2)
ax.set_yticks([])
ax.set_ylim(-0.25, 4.1)
ax.set_xlim(-5.2, 10.2)
ax.set_xticks([-5, 0, 5, 10])
ax.set_xlabel("Difference from frozen continuation (pp)")
ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left"]].set_visible(False)

fig.legend(handles=[
    Line2D([], [], marker="*", ls="", color=GOLD, ms=12, mec="white",
           label="sequential oracle (opportunity)"),
    Line2D([], [], marker="o", ls="", color=BLUE, ms=7, mec="white",
           label="frozen controller (strict)"),
], loc="lower center", ncol=2, frameon=False, fontsize=8,
    bbox_to_anchor=(0.55, -0.05), handletextpad=0.35, columnspacing=1.4)

OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig4_sequential_forest.{ext}", bbox_inches="tight",
                pad_inches=0.05)
print("wrote", OUT / "fig4_sequential_forest.{pdf,png}")
