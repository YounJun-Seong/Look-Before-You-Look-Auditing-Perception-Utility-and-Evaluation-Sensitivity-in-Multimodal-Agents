#!/usr/bin/env python3
"""Fig. 3 — deployable multi-action selector: accuracy/cost Pareto.

2x2: columns = dataset (MME-RealWorld-Lite, TextVQA), rows = cost axis
(extra visual tokens; added wall-clock latency, symlog). Curves are sweep
Pareto frontiers; the cross-replay oracle is the non-deployable upper bound.
OCR is free in tokens but expensive in seconds, so token-axis points that are
OCR-dominated are drawn hollow and annotated -- the two cost axes must never
be summed. Reads paper/multiaction_selector.json.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
R = json.load(open(ROOT / "paper/multiaction_selector.json"))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.8,
                     "axes.edgecolor": "#444444", "axes.labelcolor": "#111111",
                     "xtick.color": "#333333", "ytick.color": "#222222",
                     "pdf.fonttype": 42, "ps.fonttype": 42})

MULTI, BIN, CONF, SUB, GRID = "#0173B2", "#DE8F05", "#767676", "#767676", "#ECECEA"
ORACLE = "#029E73"
OCR_HEAVY = 0.25  # OCR share above this makes a token-axis point 'hollow'


def frontier(pts, x, eps=1e-9):
    """Upper-left Pareto frontier of sweep points on (cost, acc)."""
    pts = sorted(pts, key=lambda p: (p[x], -p["acc"]))
    out, best = [], -1.0
    for p in pts:
        if p["acc"] > best + eps:
            out.append(p)
            best = p["acc"]
    return out


def panel(ax, name, axis):
    sub = R["subsets"][name]
    rows, sweeps = sub["rows"], sub["sweeps"]
    ma = sweeps["multiaction_vtok" if axis == "vtok" else "multiaction_lat"]

    for sw, col, z in ((sweeps["confidence_stop_highres"], CONF, 3),
                       (sweeps["learned_binary_stop_highres"], BIN, 4),
                       (ma, MULTI, 5)):
        fr = frontier(sw, axis)
        xs, ys = [p[axis] for p in fr], [p["acc"] for p in fr]
        ax.plot(xs, ys, "-", color=col, lw=1.8, zorder=z)
        if col == MULTI and axis == "vtok":
            solid = [p for p in fr if p.get("mix", {}).get("OCR", 0) <= OCR_HEAVY]
            hollow = [p for p in fr if p.get("mix", {}).get("OCR", 0) > OCR_HEAVY]
            ax.plot([p[axis] for p in solid], [p["acc"] for p in solid], "o",
                    ms=3.0, mfc=col, mec="none", zorder=z + 1)
            ax.plot([p[axis] for p in hollow], [p["acc"] for p in hollow], "o",
                    ms=3.0, mfc="white", mec=col, mew=0.9, zorder=z + 1)
        else:
            ax.plot(xs, ys, "o", ms=2.6, color=col, mec="none", zorder=z + 1)

    stop, hi = rows["always_STOP"], rows["always_HIGHRES"]
    ocr, crop = rows["always_OCR"], rows["always_CROP"]
    orc, full = rows["oracle_cross_replay"], rows["full_reference"]

    ax.axhline(full["acc"], color=SUB, lw=0.9, ls=(0, (4, 3)), zorder=2)
    ax.plot(stop[axis], stop["acc"], "s", ms=6, mfc="#333333", mec="white", zorder=6)
    ax.plot(hi[axis], hi["acc"], "D", ms=6, mfc="#C44E52", mec="white", zorder=6)
    ax.plot(crop[axis], crop["acc"], "v", ms=6, mfc="#937860", mec="white", zorder=6)
    ax.plot(ocr[axis], ocr["acc"], "^", ms=6, mfc="#555555", mec="white", zorder=6)
    ax.plot(orc[axis], orc["acc"], "*", ms=12, mfc=ORACLE, mec="white", zorder=7)
    if axis == "vtok":
        ax.annotate(f"OCR: 0 tok, {ocr['lat']:.0f} s/state", (ocr["vtok"], ocr["acc"]),
                    xytext=(9, 5), textcoords="offset points", fontsize=6.2,
                    color="#555555", va="bottom")

    if axis == "lat":
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlim(-0.06, 900)
        ax.set_xticks([0, 1, 10, 100])
        ax.set_xticklabels(["0", "1", "10", "100"])
    ax.grid(color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.9), dpi=300)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.945, bottom=0.155,
                        hspace=0.36, wspace=0.24)
    titles = {"mme": "MME-RealWorld-Lite", "textvqa": "TextVQA"}
    for j, name in enumerate(("mme", "textvqa")):
        for i, axis in enumerate(("vtok", "lat")):
            ax = axes[i][j]
            panel(ax, name, axis)
            if i == 0:
                ax.set_title(titles[name], fontsize=10, pad=6)
                ax.set_xlabel("extra visual tokens / state", fontsize=8.5)
            else:
                ax.set_xlabel("added latency (s / state, symlog)", fontsize=8.5)
            if j == 0:
                ax.set_ylabel("accuracy (mean over replays)", fontsize=8.5)
            ax.tick_params(labelsize=7.5)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=MULTI, lw=1.8, marker="o", ms=3, label="learned multi-action (ours)"),
        Line2D([], [], color=BIN, lw=1.8, marker="o", ms=3, label="learned binary STOP/HIGHRES"),
        Line2D([], [], color=CONF, lw=1.8, marker="o", ms=3, label="confidence STOP/HIGHRES"),
        Line2D([], [], color=MULTI, lw=0, marker="o", ms=4.5, mfc="white", mec=MULTI,
               label="OCR-heavy mix (cheap in tokens, costly in seconds)"),
        Line2D([], [], lw=0, marker="s", ms=6, mfc="#333333", mec="white", label="always STOP"),
        Line2D([], [], lw=0, marker="D", ms=6, mfc="#C44E52", mec="white", label="always HIGHRES"),
        Line2D([], [], lw=0, marker="v", ms=6, mfc="#937860", mec="white", label="always CROP (uniform grid)"),
        Line2D([], [], lw=0, marker="^", ms=6, mfc="#555555", mec="white", label="always OCR"),
        Line2D([], [], lw=0, marker="*", ms=11, mfc=ORACLE, mec="white", label="cross-replay oracle (bound)"),
        Line2D([], [], color=SUB, lw=0.9, ls=(0, (4, 3)), label="FULL reference"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=5.9,
               frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=1.3,
               columnspacing=0.8, labelspacing=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_multiaction_pareto.{ext}")
    print("wrote fig3_multiaction_pareto.{png,pdf}")


if __name__ == "__main__":
    main()
