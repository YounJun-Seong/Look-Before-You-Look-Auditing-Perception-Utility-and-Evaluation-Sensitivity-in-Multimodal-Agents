#!/usr/bin/env python3
"""Fig. 2 — multi-action vs strong state-independent mixtures (paired deltas).

(a) TextVQA, realizability: frozen multi-action minus the DEV-FITTED deployable
    mixture at matched budgets — positive but every CI includes zero.
(b) VQAv2, measurement validity: multi-action minus the CROSS-FITTED mixture
    under strict vs lenient scoring of the same generations — the significant
    strict reversal disappears and changes sign under lenient rescoring.

Shared y-axis across panels (honest cross-panel comparison). Reads
paper/deployable_mixture_textvqa.json and paper/randomized_mixture_baseline.json.
Writes paper/figures/fig2_policy_comparison.{pdf,png}.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
PROS = json.load(open(ROOT / "paper/prospective_eval.json"))["results"]
MIX = json.load(open(ROOT / "paper/randomized_mixture_baseline.json"))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.8,
                     "axes.edgecolor": "#444444", "axes.labelcolor": "#111111",
                     "xtick.color": "#333333", "ytick.color": "#222222",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
BLUE, LIGHT, SUB, GRID = "#0173B2", "#7FB3D5", "#767676", "#ECECEA"

XLAB = ["¼×", "½×", "1×"]
X = [0, 1, 2]


def series(rows, dkey, ckey):
    d = [100 * r[dkey] for r in rows]
    lo = [100 * r[ckey][0] for r in rows]
    hi = [100 * r[ckey][1] for r in rows]
    return d, lo, hi


def draw(ax, x, d, lo, hi, color, mfc=None):
    for xi, di, l, h in zip(x, d, lo, hi):
        ax.plot([xi, xi], [l, h], color=color, lw=1.6, solid_capstyle="round",
                zorder=3)
        ax.plot([xi - 0.06, xi + 0.06], [l, l], color=color, lw=1.2, zorder=3)
        ax.plot([xi - 0.06, xi + 0.06], [h, h], color=color, lw=1.2, zorder=3)
    ax.scatter(x, d, s=42, color=mfc or color, edgecolor=color, linewidth=1.2,
               zorder=4)


fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05), dpi=300, sharey=True)
fig.subplots_adjust(left=0.115, right=0.985, top=0.835, bottom=0.19, wspace=0.08)

# (a) TextVQA — prospective n=600, strict + lenient
ax = axes[0]
rows_s = [PROS["strict"][k] for k in ("0.25", "0.5", "1.0")]
rows_l = [PROS["lenient"][k] for k in ("0.25", "0.5", "1.0")]
ds_a, lo_a, hi_a = series(rows_s, "sel_minus_mix", "ci95")
dl_a, lol_a, hil_a = series(rows_l, "sel_minus_mix", "ci95")
off = 0.13
draw(ax, [x - off for x in X], ds_a, lo_a, hi_a, BLUE)
draw(ax, [x + off for x in X], dl_a, lol_a, hil_a, LIGHT, mfc="white")
ax.set_title("(a) TextVQA, prospective (n=600)\nfrozen selector vs dev-fitted mixture",
             fontsize=9.5)

# (b) VQAv2 — cross-fitted mixture, strict vs lenient
ax = axes[1]
ds, los, his = series(MIX["vqav2/strict"], "multi_minus_mixture_cf", "ci95_cf")
dl, lol, hil = series(MIX["vqav2/lenient"], "multi_minus_mixture_cf", "ci95_cf")
off = 0.13
draw(ax, [x - off for x in X], ds, los, his, BLUE)
draw(ax, [x + off for x in X], dl, lol, hil, LIGHT, mfc="white")
ax.set_title("(b) VQAv2, post-hoc scorer sensitivity\nadaptive allocation vs cross-fitted mixture", fontsize=9.5)

for ax in axes:
    ax.axhline(0, color="#222222", lw=0.9, zorder=2)
    ax.set_xticks(X, XLAB)
    ax.set_xlim(-0.5, 2.5)
    ax.set_xlabel("Budget (× HighRes cost)")
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Adaptive allocation − state-independent\nmixture (pp)")

axes[1].legend(handles=[
    Line2D([], [], marker="o", ls="", color=BLUE, ms=6, label="strict"),
    Line2D([], [], marker="o", ls="", mfc="white", color=LIGHT, ms=6,
           mew=1.2, label="lenient (same generations)")],
    loc="lower right", frameon=False, fontsize=8.2)

OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig2_policy_comparison.{ext}")
print("wrote", OUT / "fig2_policy_comparison.{pdf,png}")
