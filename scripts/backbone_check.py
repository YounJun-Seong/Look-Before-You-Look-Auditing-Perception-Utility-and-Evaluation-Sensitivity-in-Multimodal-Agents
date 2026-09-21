#!/usr/bin/env python
"""Second-backbone replication of the confidence/perceptual-need relationship.

The paper's central finding — answer confidence ranks HELP states well on
TextVQA (ROC 0.796) and below chance on VQAv2 (0.423) — comes from one reader
(Qwen2.5-VL-7B).  This script asks the one question a second frozen backbone
can answer cheaply: is the confidence→perceptual-utility relationship
regime-dependent there too?

For each backbone and regime: HELP labels are regenerated from that backbone's
own STOP vs HIGHRES rollouts (two replays, replicate-consistent), confidence is
that backbone's mean token log-probability of the STOP answer, and we report
ROC-AUC, PR-AUC/prevalence and lift@1× with bootstrap CIs.  No training, no
PPU gate (v2.2 was fitted to Qwen traces and is not transferred).

Same lockbox state ids, same pixel views.  Token counts in the log are Qwen
28-px accounting; they are NOT the second backbone's visual token count and are
not reported here.

Outputs paper/backbone_check.{json,md} and paper/figures/fig1_backbones.{png,pdf}.
"""
from __future__ import annotations
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
THR = 0.5
BACKBONES = {
    "Qwen2.5-VL-7B": {"textvqa": "logs/lockbox_textvqa.jsonl", "vqav2": "logs/lockbox_vqav2.jsonl"},
    "Phi-4-MM-5.6B": {"textvqa": "logs/lockbox_textvqa_phi4.jsonl", "vqav2": "logs/lockbox_vqav2_phi4.jsonl"},
    # VQAv2 at coarse 64-tok-equiv was near-saturated for Phi-4 (STOP 76% vs HIGHRES 77%,
    # HELP prevalence 4%), so the coarse view was tightened to 16-tok-equiv (HIGHRES unchanged).
    "Phi-4-MM-5.6B (coarse 16)": {"vqav2": "logs/lockbox_vqav2_phi4_c16.jsonl"},
    # causal control for the format confound: same Qwen, same states, one-word answer prompt
    "Qwen2.5-VL-7B (one-word prompt)": {"vqav2": "logs/lockbox_vqav2_qwen_oneword.jsonl"},
}


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0: return "HELP"
    if d1 < 0 and d2 < 0: return "HARM"
    if d1 == 0 and d2 == 0: return "NEUTRAL"
    return "AMBIGUOUS"


def load(log):
    by = defaultdict(dict)
    for line in open(ROOT / log):
        r = json.loads(line)
        if r.get("type") == "rollout":
            by[r["sample_id"]][(r["action"]["type"], r["replay"])] = r
    out = []
    for sid, d in by.items():
        if not all(k in d for k in [("STOP", 1), ("STOP", 2), ("HIGHRES", 1), ("HIGHRES", 2)]):
            continue
        s1, s2 = d[("STOP", 1)]["score_after"], d[("STOP", 2)]["score_after"]
        x1, x2 = d[("HIGHRES", 1)]["score_after"], d[("HIGHRES", 2)]["score_after"]
        out.append({"state_id": sid, "label": four_way(s1, s2, x1, x2),
                    "stop": (s1 + s2) / 2, "hi": (x1 + x2) / 2,
                    "mean_logprob": float(np.mean(
                        [d[("STOP", r)]["confidence"]["mean_logprob"] for r in (1, 2)]))})
    return out


rng = np.random.default_rng(20260822)
res = {}
for bb, logs in BACKBONES.items():
    res[bb] = {}
    for src, log in logs.items():
        if not (ROOT / log).exists():
            continue
        rows = load(log)
        cnt = Counter(r["label"] for r in rows)
        use = [r for r in rows if r["label"] != "AMBIGUOUS"]
        y = np.array([r["label"] == "HELP" for r in use], int)
        # score: higher = less confident = "look"
        s = -np.array([r["mean_logprob"] for r in use])
        prev = y.mean()
        k = max(1, int(round(prev * len(y))))
        lift = (y[np.argsort(-s)[:k]].sum() / max(y.sum(), 1)) / (k / len(y))
        vals = []
        for _ in range(4000):
            i = rng.integers(0, len(y), len(y))
            if y[i].sum() == 0: continue
            kk = max(1, int(round(prev * len(i))))
            vals.append((y[i][np.argsort(-s[i])[:kk]].sum() / y[i].sum()) / (kk / len(i)))
        # ROC bootstrap CI too
        rocs = []
        for _ in range(4000):
            i = rng.integers(0, len(y), len(y))
            if 0 < y[i].sum() < len(i): rocs.append(roc_auc_score(y[i], s[i]))
        # deployment view on the FULL population (AMBIGUOUS included), confidence selector
        stop_all = np.array([r["stop"] for r in rows]); hi_all = np.array([r["hi"] for r in rows])
        s_all = -np.array([r["mean_logprob"] for r in rows])
        budgets = {}
        for b in (0.05, 0.10, 0.15, 0.20):
            kb = max(1, int(round(b * len(rows))))
            pick = np.zeros(len(rows), bool); pick[np.argsort(-s_all)[:kb]] = True
            budgets[f"{int(b*100)}%"] = {
                "confidence": float(np.where(pick, hi_all, stop_all).mean()),
                "random_expected": float((1 - kb / len(rows)) * stop_all.mean() + (kb / len(rows)) * hi_all.mean())}
        res[bb][src] = {
            "budgets_fullpop": budgets,
            "n_states": len(rows), "labels": dict(cnt), "n_eval": len(use), "prevalence": float(prev),
            "always_stop": float(np.mean([r["stop"] for r in rows])),
            "always_highres": float(np.mean([r["hi"] for r in rows])),
            "roc": float(roc_auc_score(y, s)), "roc_ci": [float(v) for v in np.quantile(rocs, [0.025, 0.975])],
            "pr_over_prev": float(average_precision_score(y, s) / prev),
            "lift": float(lift), "lift_ci": [float(v) for v in np.quantile(vals, [0.025, 0.975])],
        }

json.dump(res, open(ROOT / "paper/backbone_check.json", "w"), indent=1)
L = ["# Confidence → perceptual-need relationship across backbones", "",
     "Confidence-based uncertainty (−mean token log-prob of the coarse STOP answer) ranking "
     "replicate-consistent HELP states. Labels are regenerated per backbone from its own "
     "STOP/HIGHRES rollouts on the same lockbox state ids and pixel views. No training.", "",
     "| backbone | regime | n (eval) | prev. | STOP acc | HIGHRES acc | ROC [95% CI] | PR/prev | lift@1× [95% CI] |",
     "|---|---|--:|--:|--:|--:|:--:|--:|:--:|"]
for bb, d in res.items():
    for src, v in d.items():
        L.append(f"| {bb} | {src} | {v['n_states']} ({v['n_eval']}) | {100*v['prevalence']:.1f}% | "
                 f"{100*v['always_stop']:.1f}% | {100*v['always_highres']:.1f}% | "
                 f"**{v['roc']:.3f}** [{v['roc_ci'][0]:.2f}, {v['roc_ci'][1]:.2f}] | {v['pr_over_prev']:.2f}× | "
                 f"**{v['lift']:.2f}×** [{v['lift_ci'][0]:.2f}, {v['lift_ci'][1]:.2f}] |")
L.append("")
L += ["## Full-population LOOK budgets, confidence selector (AMBIGUOUS included)", "",
      "| backbone | regime | always-STOP | 5% | 10% | 15% | 20% | always-HIGHRES |", "|---|---|--:|--:|--:|--:|--:|--:|"]
for bb, d in res.items():
    for src, v in d.items():
        cells = " | ".join(f"{100*v['budgets_fullpop'][b]['confidence']:.1f}% (rand {100*v['budgets_fullpop'][b]['random_expected']:.1f})"
                           for b in ("5%", "10%", "15%", "20%"))
        L.append(f"| {bb} | {src} | {100*v['always_stop']:.1f}% | {cells} | {100*v['always_highres']:.1f}% |")
L.append("")
for bb, d in res.items():
    if len(d) == 2:
        gap = d["textvqa"]["roc"] - d["vqav2"]["roc"]
        L.append(f"- {bb}: ROC gap TextVQA − VQAv2 = **{gap:+.3f}**; "
                 f"VQAv2 confidence {'below' if d['vqav2']['roc'] < 0.5 else 'above'} chance, "
                 f"lift {'<' if d['vqav2']['lift'] < 1 else '≥'} 1×.")
open(ROOT / "paper/backbone_check.md", "w").write("\n".join(L))
print("\n".join(L))

# ------------------------------------------------------------ figure
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.8,
                     "axes.edgecolor": "#444444", "pdf.fonttype": 42})
COL = {"Qwen2.5-VL-7B": "#DE8F05", "Phi-4-MM-5.6B": "#8C4A9E", "Phi-4-MM-5.6B (coarse 16)": "#2A9D8F", "Qwen2.5-VL-7B (one-word prompt)": "#B8860B"}
MK = {"Qwen2.5-VL-7B": "o", "Phi-4-MM-5.6B": "D", "Phi-4-MM-5.6B (coarse 16)": "^", "Qwen2.5-VL-7B (one-word prompt)": "s"}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), dpi=300)
fig.subplots_adjust(left=0.13, right=0.985, top=0.78, bottom=0.23, wspace=0.32)
ys = {"textvqa": 1.0, "vqav2": 0.0}
for ax, key, ref, xlab, xlim, ticks, rl in (
        (axes[0], "roc", 0.5, "ROC-AUC of confidence-based uncertainty", (0.25, 0.95), [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], "chance"),
        (axes[1], "lift", 1.0, "Lift@1× prevalence (95% CI)", (0.0, 8.2), [0, 1, 2, 3, 4, 5, 6, 7, 8], "random = 1×")):
    for bi, (bb, d) in enumerate(res.items()):
        off = {0: 0.33, 1: 0.11, 2: -0.11, 3: -0.33}[bi]
        for src, v in d.items():
            y = ys[src] + off
            lo, hi = v[key + "_ci"]
            ax.plot([lo, hi], [y, y], color=COL[bb], lw=2.0, alpha=0.3, solid_capstyle="round", zorder=3)
            ax.plot(v[key], y, MK[bb], ms=7, mfc=COL[bb], mec="white", mew=1.0, zorder=5)
            ax.text(v[key], y + 0.10, f"{v[key]:.3f}" if key == "roc" else f"{v[key]:.2f}×",
                    ha="center", va="bottom", fontsize=7.0, color=COL[bb], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    ax.axvline(ref, color="#555555", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(ref, 1.62, rl, ha="center", va="top", fontsize=7.5, color="#767676")
    ax.set_yticks([1.0, 0.0]); ax.set_yticklabels(["TextVQA", "VQAv2"])
    ax.set_ylim(-0.55, 1.75); ax.set_xlim(*xlim); ax.set_xticks(ticks)
    ax.tick_params(axis="x", labelsize=8.5); ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlab, fontsize=9.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for gx in ticks: ax.axvline(gx, color="#ECECEA", lw=0.7, zorder=0)
axes[0].set_title("(a) Discrimination", fontsize=9.5, loc="left", pad=6)
axes[1].set_title("(b) Selectivity at matched LOOK rate", fontsize=9.5, loc="left", pad=6)
fig.legend(handles=[Line2D([], [], marker=MK[b], ls="", ms=6.5, mfc=COL[b], mec="white", label=b) for b in res],
           loc="upper center", ncol=4, frameon=False, fontsize=7.6, bbox_to_anchor=(0.5, 1.0))
for ext in ("png", "pdf"):
    fig.savefig(ROOT / f"paper/figures/fig1_backbones.{ext}")
print("wrote paper/backbone_check.{md,json}, paper/figures/fig1_backbones.{png,pdf}")
