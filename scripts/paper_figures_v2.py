"""Paper figures, revised after the 2026-08-22 review experiments.

Fig. 1  fig1_format_control   Confidence ROC-AUC per regime x backbone. The VQAv2
        row walks through the diagnosis: original strict 0.42 -> same generations,
        format-tolerant scoring 0.66 -> one-word prompt intervention 0.72 -> Phi-4
        cross-backbone 0.81.
Fig. 2  fig2_recovered_gain   Recovered share of the always-HIGHRES gain vs LOOK
        rate for a confidence selector (full population). Normalised so backbone
        accuracy and coarse-budget differences cancel. Vertical 2-panel, single
        column.
Supp.   figS_lookrate_abs     The absolute-accuracy version of Fig. 2.

Reads paper/backbone_check.json and paper/format_confound.json.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/figures"
R = json.load(open(ROOT / "paper/backbone_check.json"))
FC = json.load(open(ROOT / "paper/format_confound.json"))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.8,
                     "axes.edgecolor": "#444444", "axes.labelcolor": "#111111",
                     "xtick.color": "#333333", "ytick.color": "#222222",
                     "pdf.fonttype": 42, "ps.fonttype": 42})
QWEN, PHI, LEN, SUB, GRID = "#0173B2", "#8C4A9E", "#7FB3D5", "#767676", "#ECECEA"

Q, P = "Qwen2.5-VL-7B", "Phi-4-MM-5.6B"
Q1, P16 = "Qwen2.5-VL-7B (one-word prompt)", "Phi-4-MM-5.6B (coarse 16)"
BUDGETS = (5, 10, 15, 20)


# ------------------------------------------------------------------ Fig. 1
def fig1():
    """Final evidence structure (not discovery chronology): sealed anomaly, Phi-4,
    and the prospective lockbox with scorer-only vs prompt-control corrections."""
    fig, ax = plt.subplots(figsize=(5.0, 3.5), dpi=300)
    fig.subplots_adjust(left=0.13, right=0.995, top=0.88, bottom=0.15)
    FS = {k: FC[k] for k in ("FRESH Qwen original / vqav2", "FRESH Qwen one-word / vqav2",
                             "FRESH Qwen original / textvqa", "FRESH Qwen one-word / textvqa")}

    def point(roc, ci, y, col, mk, label, ms=7.0):
        ax.plot(ci, [y, y], color=col, lw=2.2, alpha=0.26, solid_capstyle="round", zorder=3)
        ax.plot(roc, y, mk, ms=ms, mfc=col, mec="white", mew=1.0, zorder=5)
        ax.text(roc, y + 0.10, f"{roc:.3f}", ha="center", va="bottom", fontsize=7.4, color=col, fontweight="bold")
        ax.text(0.975, y, label, ha="left", va="center", fontsize=6.8, color=col)

    # ---- TextVQA (top)
    yt = [3.05, 2.75, 2.45, 2.15]
    point(R[Q]["textvqa"]["roc"], R[Q]["textvqa"]["roc_ci"], yt[0], QWEN, "o", "sealed / original")
    e = FS["FRESH Qwen original / textvqa"]["strict"]
    point(e["conf_roc"], e["conf_roc_ci"], yt[1], QWEN, "s", "prospective / original", ms=6.3)
    e = FS["FRESH Qwen one-word / textvqa"]["strict"]
    point(e["conf_roc"], e["conf_roc_ci"], yt[2], QWEN, "s", "prospective / one-word", ms=6.3)
    point(R[P]["textvqa"]["roc"], R[P]["textvqa"]["roc_ci"], yt[3], PHI, "D", "Phi-4", ms=6.3)

    # ---- VQAv2 (bottom): Qwen causal story first, cross-reader validation last
    yv = [1.45, 1.12, 0.79, 0.46, 0.13]
    o = FS["FRESH Qwen original / vqav2"]["strict"]
    l = FS["FRESH Qwen original / vqav2"]["lenient"]
    w = FS["FRESH Qwen one-word / vqav2"]["strict"]
    point(R[Q]["vqav2"]["roc"], R[Q]["vqav2"]["roc_ci"], yv[0], QWEN, "o", "sealed / original")
    point(o["conf_roc"], o["conf_roc_ci"], yv[1], QWEN, "s", "prospective / original", ms=6.3)
    point(l["conf_roc"], l["conf_roc_ci"], yv[2], QWEN, "s", "prospective / original + lenient", ms=6.3)
    point(w["conf_roc"], w["conf_roc_ci"], yv[3], QWEN, "s", "prospective / one-word", ms=6.3)
    point(R[P]["vqav2"]["roc"], R[P]["vqav2"]["roc_ci"], yv[4], PHI, "D", "Phi-4 / original views", ms=6.3)

    ax.axvline(0.5, color="#555555", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(0.5, 3.32, "chance", ha="center", va="bottom", fontsize=6.8, color="#A0A0A0")
    ax.axhline(1.80, xmin=0.0, xmax=1.0, color="#E3E3E3", lw=0.8, zorder=0)
    ax.set_yticks([sum(yt) / 4, sum(yv) / 5]); ax.set_yticklabels(["TextVQA", "VQAv2"], fontsize=10)
    ax.set_ylim(-0.12, 3.32)
    ax.set_xlim(0.30, 1.26); ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax.tick_params(axis="x", labelsize=8.5); ax.tick_params(axis="y", length=0)
    # x-label with a synthetic small-caps "Help" to match \textsc{Help} in the paper
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
    parts = [TextArea("Confidence ROC–AUC for ", textprops=dict(fontsize=9.5, color="#111111")),
             TextArea("H", textprops=dict(fontsize=9.5, color="#111111")),
             TextArea("ELP", textprops=dict(fontsize=7.4, color="#111111")),
             TextArea(" (95% CI)", textprops=dict(fontsize=9.5, color="#111111"))]
    box = AnchoredOffsetbox(loc="upper center", child=HPacker(children=parts, align="baseline", pad=0, sep=0),
                            bbox_to_anchor=(0.37, -0.075), bbox_transform=ax.transAxes, frameon=False, borderpad=0)
    ax.add_artist(box)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_bounds(0.3, 0.96)
    for gx in ax.get_xticks(): ax.axvline(gx, color=GRID, lw=0.7, zorder=0)
    handles = [Line2D([], [], marker="o", ls="", ms=6.5, mfc=QWEN, mec="white", label="Qwen, sealed diagnostic"),
               Line2D([], [], marker="s", ls="", ms=6, mfc=QWEN, mec="white", label="Qwen, prospective format test"),
               Line2D([], [], marker="D", ls="", ms=6, mfc=PHI, mec="white", label="Phi-4-multimodal")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7.4, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.4, columnspacing=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig1_format_control.{ext}", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


# ------------------------------------------------------------------ helpers
def curves(e):
    s, h = 100 * e["always_stop"], 100 * e["always_highres"]
    conf = [s] + [100 * e["budgets_fullpop"][f"{b}%"]["confidence"] for b in BUDGETS]
    rand = [s] + [100 * e["budgets_fullpop"][f"{b}%"]["random_expected"] for b in BUDGETS]
    return s, h, conf, rand


PANELS = [("(a) TextVQA", {Q: (R[Q]["textvqa"], "192→512"), P: (R[P]["textvqa"], "192→512")}),
          ("(b) VQAv2", {Q: (R[Q1]["vqav2"], "64→192, one-word prompt"),
                                                   P: (R[P16]["vqav2"], "16→192, native prompt")})]
LEGEND = [Line2D([], [], color=QWEN, lw=1.9, marker="o", ms=5.5, mec="white", label="Qwen2.5-VL-7B, confidence selector"),
          Line2D([], [], color=PHI, lw=1.9, marker="D", ms=5, mec="white", label="Phi-4-multimodal, confidence selector"),
          Line2D([], [], color="#777777", lw=1.0, ls=(0, (2, 2)), label="random allocation (expected)")]


# ------------------------------------------------------------------ Fig. 2
def fig2():
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.8), dpi=300, sharex=True)
    fig.subplots_adjust(left=0.17, right=0.99, top=0.85, bottom=0.10, hspace=0.30)
    rates = [0] + list(BUDGETS)
    for ax, (title, d) in zip(axes, PANELS):
        ax.plot(rates, rates, ls=(0, (2, 2)), color="#777777", lw=1.0, zorder=2)
        for bb, col, mk in ((Q, QWEN, "o"), (P, PHI, "D")):
            e, cond = d[bb]
            s, h, conf, _ = curves(e)
            rec = [100 * (c - s) / (h - s) for c in conf]
            ax.plot(rates, rec, "-", color=col, lw=1.9, zorder=3)
            ax.plot(rates[1:], rec[1:], mk, ms=5.5, mfc=col, mec="white", mew=1.0, zorder=5)
            ax.text(20.6, rec[-1], f"{rec[-1]:.0f}% recovered", fontsize=7.2, color=col, ha="left", va="center", fontweight="bold")
        ax.text(0.02, 0.97, "\n".join(f"{'Qwen' if bb == Q else 'Phi-4'}: {d[bb][1]}" for bb in (Q, P)),
                transform=ax.transAxes, fontsize=6.4, color=SUB, ha="left", va="top", linespacing=1.25)
        ax.set_title(title, fontsize=9, loc="left", pad=4)
        ax.set_xlim(-0.5, 30.5); ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_ylim(-2, 72); ax.set_yticks([0, 20, 40, 60])
        ax.tick_params(labelsize=8)
        ax.set_ylabel("Recovered H\u1d35\u1d33\u1d34R\u1d31\u02e2 gain (%)" if False else "Recovered HighRes gain (%)", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    axes[1].set_xlabel("Look rate (% of states sent to HighRes)", fontsize=8.5)
    fig.legend(handles=LEGEND, loc="upper center", ncol=1, frameon=False, fontsize=7.4,
               bbox_to_anchor=(0.56, 1.0), handletextpad=0.5, labelspacing=0.25)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_recovered_gain.{ext}")
    plt.close(fig)


# ------------------------------------------------------------------ Supp.
def figS():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), dpi=300)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.80, bottom=0.20, wspace=0.22)
    rates = [0] + list(BUDGETS)
    for ax, (title, d) in zip(axes, PANELS):
        for bb, col, mk in ((Q, QWEN, "o"), (P, PHI, "D")):
            e, cond = d[bb]
            s, h, conf, rand = curves(e)
            name = "Qwen" if bb == Q else "Phi-4"
            ax.plot(rates, rand, ls=(0, (2, 2)), color=col, lw=1.0, alpha=0.6, zorder=2)
            ax.plot(rates, conf, "-", color=col, lw=1.9, zorder=3)
            ax.plot(rates[1:], conf[1:], mk, ms=6, mfc=col, mec="white", mew=1.0, zorder=5)
            ax.plot(0, s, "s", ms=5, mfc=col, mec="white", mew=0.8, zorder=5)
            ax.axhline(h, color=col, lw=0.9, ls=(0, (5, 3)), alpha=0.7, zorder=1)
            if bb == Q:
                ax.text(0.2, h + 0.25, f"{name} HIGHRES {h:.1f}%", fontsize=6.8, color=col, va="bottom", ha="left")
            else:
                ax.text(25.3, h - 0.25, f"{name} HIGHRES {h:.1f}%", fontsize=6.8, color=col, va="top", ha="right")
            frac = (conf[4] - s) / (h - s)
            ax.text(20.4, conf[4], f"{frac:.0%} of gain", fontsize=7, color=col, ha="left", va="center", fontweight="bold")
        note = "\n".join(f"{'Qwen' if bb == Q else 'Phi-4'}: {d[bb][1]}" for bb in (Q, P))
        if title.startswith("(a)"):
            ax.text(0.02, 0.56, note, transform=ax.transAxes, fontsize=6.4, color=SUB, ha="left", va="center", linespacing=1.25)
        else:
            ax.text(0.98, 0.36, note, transform=ax.transAxes, fontsize=6.4, color=SUB, ha="right", va="center", linespacing=1.25)
        ax.set_title(title, fontsize=9.5, loc="left", pad=5)
        ax.set_xlim(-0.5, 25.5); ax.set_xticks([0, 5, 10, 15, 20])
        ax.tick_params(labelsize=8.5)
        ax.set_xlabel("LOOK rate (% of states sent to HIGHRES)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    axes[0].set_ylabel("Task accuracy (%)", fontsize=9.5)
    fig.legend(handles=LEGEND, loc="upper center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0),
               handletextpad=0.5, columnspacing=1.4)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figS_lookrate_abs.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); figS()
    for old in ("fig2_lookrate.png", "fig2_lookrate.pdf"):
        (OUT / old).unlink(missing_ok=True)
    print("saved fig1_format_control, fig2_recovered_gain, figS_lookrate_abs ->", OUT)
