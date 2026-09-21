#!/usr/bin/env python3
"""(iv) Supp. fig — accuracy-cost Pareto on BOTH cost axes (zero inference).

Same frozen held-out operating points as the token-budget tables
(paper/heldout_multiaction.json), plotted against (left) extra visual tokens
and (right) added wall-clock latency (generation + OCR tool time). The two
axes are never summed; the figure shows that the optimal policy class depends
on which resource the agent is constrained by.

Writes paper/figures/figS_latency_pareto.{pdf,png}.
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

fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6), dpi=300)
fig.subplots_adjust(left=0.085, right=0.99, top=0.95, bottom=0.14,
                    hspace=0.42, wspace=0.22)

for row, ds in enumerate(("vqav2", "textvqa")):
    s = D[ds]["scorers"]["strict"]
    rows, sw = s["rows"], s["sweeps"]
    for col, axis in enumerate(("vtok", "lat")):
        ax = axes[row][col]
        for name, col_, ls, lw in (("multiaction_vtok" if axis == "vtok" else "multiaction_lat",
                                    BLUE, "-", 2.2),
                                   ("v22_gate", PURPLE, "--", 1.3),
                                   ("confidence", LIGHT, "-", 1.3),
                                   ("random_allocation", SUB, ":", 1.2)):
            pts = sorted(((p[axis], p["acc"]) for p in sw[name]), key=lambda t: t[0])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=col_,
                    lw=lw, zorder=3)
        for name, mk in (("always_STOP", "o"), ("always_OCR", "^"),
                         ("always_CROP", "s"), ("always_HIGHRES", "D")):
            r = rows[name]
            ax.scatter([r[axis]], [r["acc"]], marker=mk, s=24, color="#222222",
                       zorder=5)
            dx = 4 if name != "always_OCR" else 4
            ax.annotate(name.replace("always_", ""), (r[axis], r["acc"]),
                        textcoords="offset points", xytext=(dx, 5),
                        fontsize=6.6, color="#222222")
        o = rows["oracle_cross_replay"]
        ax.scatter([o[axis]], [o["acc"]], marker="*", s=110, color="#C99700",
                   edgecolor="#222222", linewidth=0.5, zorder=6)
        if axis == "lat":
            ax.set_xscale("symlog", linthresh=1.0)
        ax.grid(color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(f"{'VQAv2' if ds=='vqav2' else 'TextVQA'} — "
                     f"{'extra visual tokens' if axis=='vtok' else 'added latency (s, symlog)'}",
                     fontsize=9)
        if col == 0:
            ax.set_ylabel("accuracy (strict)")
for ax in axes[1]:
    ax.set_xlabel("cost per state")

fig.legend(handles=[
    Line2D([], [], color=BLUE, lw=2.2, label="frozen multi-action (λ sweep on shown axis)"),
    Line2D([], [], color=PURPLE, lw=1.3, ls="--", label="v2.2 binary gate"),
    Line2D([], [], color=LIGHT, lw=1.3, label="confidence STOP/HIGHRES"),
    Line2D([], [], color=SUB, lw=1.2, ls=":", label="random allocation"),
    Line2D([], [], marker="*", ls="", color="#C99700", ms=10, mec="#222222",
           mew=0.5, label="cross-replay oracle"),
], loc="lower center", ncol=3, frameon=False, fontsize=7.8,
    bbox_to_anchor=(0.5, -0.005))

OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"figS_latency_pareto.{ext}", bbox_inches="tight",
                pad_inches=0.05)
print("wrote", OUT / "figS_latency_pareto.{pdf,png}")
