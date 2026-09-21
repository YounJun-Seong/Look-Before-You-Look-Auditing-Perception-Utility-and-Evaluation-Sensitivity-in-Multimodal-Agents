#!/usr/bin/env python3
"""Supp. fig — budget frontier, VQAv2 strict vs lenient, IDENTICAL y-axis.

Accuracy vs mean extra visual tokens for every policy class of the prereg
evaluation: fixed actions (points), random STOP<->HIGHRES allocation (dotted),
confidence and v2.2 binary selectors (thin lines), the frozen multi-action
selector's lambda sweep (thick line), and the cross-replay oracle (star).
Vertical guides mark the 1/4x, 1/2x and 1x always-HIGHRES budgets.

Reads paper/heldout_multiaction.json (frozen numbers only — no recomputation).
Writes paper/figures/figS_budget_frontier_lenient_matched.{pdf,png}.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
D = json.load(open(ROOT / "paper/heldout_multiaction.json"))["datasets"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.8,
                     "axes.edgecolor": "#444444", "axes.labelcolor": "#111111",
                     "xtick.color": "#333333", "ytick.color": "#222222",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
BLUE, PURPLE, LIGHT, SUB, GRID = "#0173B2", "#8C4A9E", "#7FB3D5", "#767676", "#ECECEA"

PANELS = [(("vqav2", "strict"), "VQAv2 (held-out, strict)"),
          (("vqav2", "lenient"), "VQAv2 (held-out, lenient rescore)")]


def curve(sweep):
    pts = sorted(((p["vtok"], p["acc"]) for p in sweep), key=lambda t: t[0])
    return [p[0] for p in pts], [p[1] for p in pts]


fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), dpi=300)
fig.subplots_adjust(left=0.075, right=0.995, top=0.86, bottom=0.24, wspace=0.22)

for ax, ((ds, sc), title) in zip(axes, PANELS):
    s = D[ds]["scorers"][sc]
    rows, sw = s["rows"], s["sweeps"]
    hi_tok = rows["always_HIGHRES"]["vtok"]

    for frac, lab in ((0.25, "¼×"), (0.5, "½×"), (1.0, "1×")):
        ax.axvline(frac * hi_tok, color=GRID, lw=1.2, zorder=0)
        ax.text(frac * hi_tok, ax.get_ylim()[0], lab, color=SUB, fontsize=8,
                ha="center", va="bottom", transform=ax.get_xaxis_transform())

    x, y = curve(sw["random_allocation"])
    ax.plot(x, y, ":", color=SUB, lw=1.2, zorder=2)
    x, y = curve(sw["confidence"])
    ax.plot(x, y, "-", color=LIGHT, lw=1.3, zorder=3)
    x, y = curve(sw["v22_gate"])
    ax.plot(x, y, "--", color=PURPLE, lw=1.3, zorder=3)
    x, y = curve(sw["multiaction_vtok"])
    ax.plot(x, y, "-", color=BLUE, lw=2.4, zorder=4)

    for name, mk, xy, ha in (("always_STOP", "o", (6, -11), "left"),
                             ("always_OCR", "^", (6, 7), "left"),
                             ("always_CROP", "s", (5, -11), "left"),
                             ("always_HIGHRES", "D", (-7, 8), "right")):
        r = rows[name]
        ax.scatter([r["vtok"]], [r["acc"]], marker=mk, s=26, color="#222222", zorder=5)
        ax.annotate(name.replace("always_", ""), (r["vtok"], r["acc"]),
                    textcoords="offset points", xytext=xy, fontsize=7.5,
                    ha=ha, color="#222222")
    o = rows["oracle_cross_replay"]
    ax.scatter([o["vtok"]], [o["acc"]], marker="*", s=130, color="#C99700",
               edgecolor="#222222", linewidth=0.5, zorder=6)
    ax.annotate("oracle", (o["vtok"], o["acc"]), textcoords="offset points",
                xytext=(6, 2), fontsize=8, color="#7a5c00")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("mean extra visual tokens / state")
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
for ax in axes:
    ax.set_ylim(0.42, 0.90)
axes[0].set_ylabel("accuracy")

fig.legend(handles=[
    Line2D([], [], color=BLUE, lw=2.4, label="frozen multi-action (λ sweep)"),
    Line2D([], [], color=PURPLE, lw=1.3, ls="--", label="v2.2 binary gate"),
    Line2D([], [], color=LIGHT, lw=1.3, label="confidence STOP/HIGHRES"),
    Line2D([], [], color=SUB, lw=1.2, ls=":", label="random allocation"),
    Line2D([], [], marker="*", ls="", color="#C99700", ms=11, mec="#222222",
           mew=0.5, label="cross-replay oracle"),
    Line2D([], [], marker="o", ls="", color="#222222", ms=5, label="fixed actions"),
], loc="lower center", ncol=3, frameon=False, fontsize=8.2,
    bbox_to_anchor=(0.5, -0.015))

OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figS_budget_frontier_lenient_matched.{ext}")
print("wrote", OUT / "figS_budget_frontier_lenient_matched.{pdf,png}")
