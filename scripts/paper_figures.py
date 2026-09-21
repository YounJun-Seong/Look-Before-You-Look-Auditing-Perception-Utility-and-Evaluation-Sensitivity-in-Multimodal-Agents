"""Paper figures for PPU (Stage A).

Fig. 1  Confidence reversal across perceptual regimes (dumbbell, 2 panels)
Fig. 2  Accuracy–perception-cost trade-off on VQAv2 (Pareto)

All numbers are copied verbatim from paper/STAGE_A_REPORT.md §1
(fresh lockbox, single evaluation). The random-gate curve in Fig. 2 is the
analytic expectation STOP + rho * (HIGHRES - STOP); it is labelled as such.

Usage:  .venv/bin/python scripts/paper_figures.py
Output: paper/figures/fig1_confidence_reversal.{png,pdf}
        paper/figures/fig2_pareto_vqav2.{png,pdf}
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#111111",
    "xtick.color": "#333333",
    "ytick.color": "#222222",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PPU = "#0173B2"      # blue
CONF = "#DE8F05"     # orange
RAND = "#9A9A9A"
INK = "#111111"
SUB = "#767676"
GRID = "#ECECEA"

# --------------------------------------------------------------------------
# Data (STAGE_A_REPORT.md §1)
# --------------------------------------------------------------------------
REGIMES = ["TextVQA", "VQAv2"]
ROC = {"Confidence": [0.796, 0.423], "PPU gate": [0.669, 0.703]}
LIFT = {"Confidence": [3.2, 0.7], "PPU gate": [1.52, 1.77]}
LIFT_CI = {"Confidence": [(2.16, 4.05), (0.26, 1.18)],
           "PPU gate": [(0.62, 2.42), (1.20, 2.35)]}

# VQAv2 fixed-LOOK-budget table
STOP_ACC, HR_ACC, HR_TOK = 44.2, 65.2, 128
BUDGET = [0.05, 0.10, 0.15, 0.20]
TOK = [6, 12, 19, 25]
PPU_ACC = [47.2, 48.7, 50.2, 51.7]
CONF_ACC = [44.3, 44.5, 46.2, 47.2]


# --------------------------------------------------------------------------
# Fig. 1 — dumbbell plot
# --------------------------------------------------------------------------
def fig1():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), dpi=300)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.80, bottom=0.22, wspace=0.32)

    ys = {"TextVQA": 1.0, "VQAv2": 0.0}

    def panel(ax, data, ci, ref, xlab, xlim, xticks, ref_label, ci_panel):
        for reg in REGIMES:
            y = ys[reg]
            c, p = data["Confidence"][REGIMES.index(reg)], data["PPU gate"][REGIMES.index(reg)]
            # connector
            if not ci_panel:
                ax.plot([c, p], [y, y], color="#BBBBBB", lw=2.0, zorder=2,
                        solid_capstyle="round")
            # CI whiskers (lift panel only)
            if ci_panel:
                for val, (lo, hi), col, dy in (
                        (c, ci["Confidence"][REGIMES.index(reg)], CONF, 0.0),
                        (p, ci["PPU gate"][REGIMES.index(reg)], PPU, 0.0)):
                    ax.plot([lo, hi], [y, y], color=col, lw=2.0,
                            alpha=0.30, zorder=3, solid_capstyle="round")
            ax.plot(c, y, "o", ms=8.5, mfc=CONF, mec="white", mew=1.2, zorder=5)
            ax.plot(p, y, "o", ms=8.5, mfc=PPU, mec="white", mew=1.2, zorder=5)
            # value labels
            fmt = (lambda v: f"{v:.2f}×") if ci_panel else (lambda v: f"{v:.3f}")
            for val, col in ((c, CONF), (p, PPU)):
                ax.text(val, y + 0.20, fmt(val), ha="center", va="bottom",
                        fontsize=8.2, color=col, fontweight="bold", zorder=6)
        ax.axvline(ref, color="#555555", lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax.text(ref, 1.58, ref_label, ha="center", va="top", fontsize=7.5,
                color=SUB)
        ax.set_yticks([1.0, 0.0])
        ax.set_yticklabels(REGIMES, fontsize=10)
        ax.set_ylim(-0.45, 1.65)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(xlab, fontsize=9.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        for gx in xticks:
            ax.axvline(gx, color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)

    panel(axes[0], ROC, None, 0.5, "ROC-AUC (HELP vs. rest)",
          (0.30, 0.90), [0.4, 0.5, 0.6, 0.7, 0.8], "chance", False)
    panel(axes[1], LIFT, LIFT_CI, 1.0, "Lift@1× prevalence (95% CI)",
          (0.0, 4.4), [0, 1, 2, 3, 4], "random = 1×", True)

    axes[0].set_title("(a) Discrimination", fontsize=9.5, loc="left", pad=6)
    axes[1].set_title("(b) Selectivity at matched LOOK rate", fontsize=9.5,
                      loc="left", pad=6)

    handles = [Line2D([], [], marker="o", ls="", ms=7, mfc=CONF, mec="white", label="Confidence-based uncertainty"),
               Line2D([], [], marker="o", ls="", ms=7, mfc=PPU, mec="white", label="PPU gate (v2.2, frozen)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=8.8, bbox_to_anchor=(0.5, 1.0), handletextpad=0.4,
               columnspacing=1.6)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig1_confidence_reversal.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------------
# Fig. 2 — accuracy vs. perception cost, VQAv2
# --------------------------------------------------------------------------
def fig2():
    fig, ax = plt.subplots(figsize=(5.6, 3.3), dpi=300)
    fig.subplots_adjust(left=0.12, right=0.975, top=0.93, bottom=0.16)

    # reference lines
    ax.axhline(STOP_ACC, color="#666666", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.axhline(HR_ACC, color="#666666", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(126, STOP_ACC - 0.6, "Always STOP  44.2%", ha="right", va="top",
            fontsize=8, color=SUB)
    ax.text(126, HR_ACC + 0.6, "Always HIGHRES  65.2% @ 128 tok", ha="right",
            va="bottom", fontsize=8, color=SUB)

    # random gate expectation (linear interpolation STOP -> HIGHRES)
    ax.plot([0, HR_TOK], [STOP_ACC, HR_ACC], color=RAND, lw=1.2,
            ls=(0, (2, 2)), zorder=2, label="Random gate @ ρ (expected)")

    # endpoints
    ax.plot(0, STOP_ACC, "s", ms=6.5, mfc="#666666", mec="white", mew=1.0, zorder=5)
    ax.plot(HR_TOK, HR_ACC, "s", ms=6.5, mfc="#666666", mec="white", mew=1.0, zorder=5)

    # confidence gate
    ax.plot([0] + TOK, [STOP_ACC] + CONF_ACC, "-", color=CONF, lw=1.6, zorder=3)
    ax.plot(TOK, CONF_ACC, "x", color=CONF, ms=7, mew=1.8, zorder=6,
            label="Confidence-based uncertainty")

    # PPU gate
    ax.plot([0] + TOK, [STOP_ACC] + PPU_ACC, "-", color=PPU, lw=2.0, zorder=4)
    ax.plot(TOK, PPU_ACC, "o", color=PPU, ms=7, mfc=PPU, mec="white", mew=1.2,
            zorder=7, label="PPU gate (v2.2, frozen)")

    # budget labels on PPU points
    for b, t, a in zip(BUDGET, TOK, PPU_ACC):
        ax.annotate(f"{int(b*100)}%", (t, a), xytext=(-2, 7),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=7.5, color=PPU)

    # annotation at 25 tok
    gain_frac = (PPU_ACC[-1] - STOP_ACC) / (HR_ACC - STOP_ACC)
    ax.annotate(
        f"20% LOOK budget\n{TOK[-1]}/{HR_TOK} tokens = {TOK[-1]/HR_TOK:.0%} of HIGHRES cost\n"
        f"Recovers {gain_frac:.0%} of the always-HIGHRES accuracy gain",
        xy=(TOK[-1], PPU_ACC[-1]), xytext=(50, 48.3),
        fontsize=8, color=INK, ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color="#777777", lw=0.8,
                        shrinkA=0, shrinkB=4),
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CCCCCC", lw=0.6))

    ax.set_xlim(-3, 134)
    ax.set_ylim(42, 68)
    ax.set_xticks([0, 6, 12, 19, 25, 64, 128])
    ax.set_yticks([44, 48, 52, 56, 60, 64])
    ax.tick_params(labelsize=8.5)
    ax.set_xlabel("Additional visual tokens per state", fontsize=9.5)
    ax.set_ylabel("Task accuracy (%)", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.78), frameon=False,
              fontsize=8.3, handlelength=2.2)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_pareto_vqav2.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    print("saved to", OUT)
