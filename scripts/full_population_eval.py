#!/usr/bin/env python
"""Deployment-view evaluation on the FULL lockbox population, AMBIGUOUS included.

The primary gate metrics exclude replay-ambiguous states because their HELP /
HARM label is undefined.  A deployed agent cannot know which states those are,
so this script runs the frozen gate over every state and reports only what a
deployment sees: downstream accuracy and added visual tokens at fixed LOOK
budgets.  No classification metrics, no re-labelling, no refit.

Accuracy per state = mean over the two replays of the executed action's score
(STOP if not selected, HIGHRES if selected).  Random gate = mean over 2000
uniformly random selections of the same size (its expectation is the linear
interpolation between always-STOP and always-HIGHRES).

Outputs paper/full_population_eval.{json,md} and
paper/figures/fig2_fullpop_{vqav2,textvqa}.{png,pdf}.
"""
from __future__ import annotations
import json, pickle, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppu.gate_features import extract  # noqa: E402

THR = 0.5
FR = pickle.load(open(ROOT / "models/gate_frozen_v2.2_pointwise.pkl", "rb"))
DEV_CDF = FR["conf_transform"]["sorted_dev_conf"]
SETS = [("textvqa", "logs/lockbox_textvqa.jsonl"), ("vqav2", "logs/lockbox_vqav2.jsonl")]
BUDGETS = [0.05, 0.10, 0.15, 0.20]
N_RANDOM = 2000


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0: return "HELP"
    if d1 < 0 and d2 < 0: return "HARM"
    if d1 == 0 and d2 == 0: return "NEUTRAL"
    return "AMBIGUOUS"


def conf_pct(x):
    return float(1.0 - np.searchsorted(DEV_CDF, x) / len(DEV_CDF))


def load(log, source):
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
        st = d[("STOP", 1)]
        out.append({"state_id": sid, "source": source, "label": four_way(s1, s2, x1, x2),
                    "answer": st["answer_after"], "question": st["question"],
                    "answer_len_tokens": st["cost"]["tokens_out"],
                    "stop": (s1 + s2) / 2, "hi": (x1 + x2) / 2,
                    "hi_tokens": d[("HIGHRES", 1)]["cost"]["visual_tokens"],
                    "mean_logprob": float(np.mean(
                        [d[("STOP", r)]["confidence"]["mean_logprob"] for r in (1, 2)]))})
    return out


rng = np.random.default_rng(20260822)
results = {}
for source, log in SETS:
    rows = load(log, source)
    lab = Counter(r["label"] for r in rows)
    conf = np.array([conf_pct(r["mean_logprob"]) for r in rows])
    X = np.array([extract(r, c) for r, c in zip(rows, conf)], float)
    v22 = FR["model"].predict_proba(FR["scaler"].transform(X))[:, 1]
    stop = np.array([r["stop"] for r in rows]); hi = np.array([r["hi"] for r in rows])
    tok = np.array([r["hi_tokens"] for r in rows], float)
    n = len(rows)
    amb = np.array([r["label"] == "AMBIGUOUS" for r in rows])

    def run(score, b):
        k = max(1, int(round(b * n)))
        pick = np.zeros(n, bool); pick[np.argsort(-score)[:k]] = True
        return {"acc": float(np.where(pick, hi, stop).mean()),
                "tok": float(tok[pick].sum() / n),
                "amb_share_of_picks": float(amb[pick].mean())}

    def run_random(b):
        k = max(1, int(round(b * n)))
        accs, toks = [], []
        for _ in range(N_RANDOM):
            pick = np.zeros(n, bool); pick[rng.choice(n, k, replace=False)] = True
            accs.append(np.where(pick, hi, stop).mean()); toks.append(tok[pick].sum() / n)
        return {"acc": float(np.mean(accs)), "acc_sd": float(np.std(accs)),
                "tok": float(np.mean(toks)),
                "acc_expected": float((1 - k / n) * stop.mean() + (k / n) * hi.mean())}

    results[source] = {
        "n": n, "labels": dict(lab), "ambiguous_frac": float(amb.mean()),
        "always_stop": {"acc": float(stop.mean()), "tok": 0.0},
        "always_highres": {"acc": float(hi.mean()), "tok": float(tok.mean())},
        "budgets": {f"{int(b*100)}%": {"ppu_v22": run(v22, b), "confidence": run(conf, b),
                                       "random": run_random(b)} for b in BUDGETS},
        # same gates restricted to the non-ambiguous subset, for direct comparison
        "nonambiguous_reference": {
            f"{int(b*100)}%": {
                "ppu_v22": float(np.where(
                    np.isin(np.arange(n), np.argsort(-np.where(amb, -np.inf, v22))[:max(1, int(round(b * (~amb).sum())))]),
                    hi, stop)[~amb].mean()),
            } for b in BUDGETS},
    }

out_json = ROOT / "paper/full_population_eval.json"
out_md = ROOT / "paper/full_population_eval.md"
json.dump(results, open(out_json, "w"), indent=1)

L = ["# Full-population deployment evaluation (AMBIGUOUS included)", "",
     "Frozen gate v2.2 and the confidence selector run over every lockbox state; "
     "replay-ambiguous states are neither removed nor relabelled. Accuracy is "
     "replay-averaged task accuracy of the executed action. Random = mean of 2000 "
     "uniform selections of the same size (expected value = linear interpolation).", ""]
for source in ("textvqa", "vqav2"):
    R = results[source]
    lab = R["labels"]
    L += [f"## {source.upper()}  n={R['n']}  "
          f"(HELP {lab.get('HELP',0)} / HARM {lab.get('HARM',0)} / NEUTRAL {lab.get('NEUTRAL',0)} / "
          f"AMBIGUOUS {lab.get('AMBIGUOUS',0)} = {100*R['ambiguous_frac']:.1f}%)", "",
          f"always-STOP {100*R['always_stop']['acc']:.1f}% @ 0 tok · "
          f"always-HIGHRES {100*R['always_highres']['acc']:.1f}% @ {R['always_highres']['tok']:.0f} tok/state", "",
          "| LOOK budget | PPU v2.2 acc | confidence acc | random acc (±sd) | tok/state (PPU) | AMBIG share of PPU picks |",
          "|---|--:|--:|--:|--:|--:|"]
    for b, v in R["budgets"].items():
        L.append(f"| {b} | **{100*v['ppu_v22']['acc']:.1f}%** | {100*v['confidence']['acc']:.1f}% | "
                 f"{100*v['random']['acc']:.1f}% (±{100*v['random']['acc_sd']:.1f}) | "
                 f"{v['ppu_v22']['tok']:.0f} | {100*v['ppu_v22']['amb_share_of_picks']:.0f}% |")
    L.append("")
open(out_md, "w").write("\n".join(L))
print("\n".join(L))

# ---------------------------------------------------------------- figures
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.8,
                     "axes.edgecolor": "#444444", "pdf.fonttype": 42})
PPU, CONF, RAND, SUB, GRID = "#0173B2", "#DE8F05", "#9A9A9A", "#767676", "#ECECEA"
FIG = ROOT / "paper/figures"
for source in ("vqav2", "textvqa"):
    R = results[source]
    s_acc, h_acc, h_tok = 100 * R["always_stop"]["acc"], 100 * R["always_highres"]["acc"], R["always_highres"]["tok"]
    fig, ax = plt.subplots(figsize=(5.6, 3.3), dpi=300)
    fig.subplots_adjust(left=0.12, right=0.975, top=0.90, bottom=0.16)
    ax.axhline(s_acc, color="#666666", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.axhline(h_acc, color="#666666", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(h_tok * 0.985, s_acc - 0.5, f"Always STOP  {s_acc:.1f}%", ha="right", va="top", fontsize=8, color=SUB)
    ax.text(h_tok * 0.985, h_acc + 0.5, f"Always HIGHRES  {h_acc:.1f}% @ {h_tok:.0f} tok", ha="right", va="bottom", fontsize=8, color=SUB)
    ax.plot([0, h_tok], [s_acc, h_acc], color=RAND, lw=1.2, ls=(0, (2, 2)), zorder=2, label="Random gate @ ρ (expected)")
    ax.plot(0, s_acc, "s", ms=6.5, mfc="#666666", mec="white", zorder=5)
    ax.plot(h_tok, h_acc, "s", ms=6.5, mfc="#666666", mec="white", zorder=5)
    for key, col, mk, lab in (("confidence", CONF, "x", "Confidence-based uncertainty"),
                              ("ppu_v22", PPU, "o", "PPU gate (v2.2, frozen)")):
        xs = [0] + [R["budgets"][b][key]["tok"] for b in R["budgets"]]
        ys = [s_acc] + [100 * R["budgets"][b][key]["acc"] for b in R["budgets"]]
        ax.plot(xs, ys, "-", color=col, lw=2.0 if key == "ppu_v22" else 1.6, zorder=3)
        ax.plot(xs[1:], ys[1:], mk, color=col, ms=7, mfc=col if mk == "o" else None,
                mec="white" if mk == "o" else col, mew=1.2 if mk == "o" else 1.8, zorder=6, label=lab)
    for b, x, yv in zip(R["budgets"], xs[1:], ys[1:]):
        ax.annotate(b, (x, yv), xytext=(-2, 7), textcoords="offset points", ha="right", fontsize=7.5, color=PPU)
    ax.set_xlim(-h_tok * 0.02, h_tok * 1.05)
    lo, hi_y = min(s_acc, min(ys)) - 2, h_acc + 3
    ax.set_ylim(lo, hi_y)
    ax.set_xlabel("Additional visual tokens per state", fontsize=9.5)
    ax.set_ylabel("Task accuracy (%)", fontsize=9.5)
    ax.set_title(f"{source.upper() if source=='vqav2' else 'TextVQA'} — full population incl. "
                 f"{100*R['ambiguous_frac']:.0f}% replay-ambiguous states (n={R['n']})",
                 fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
    ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.78), frameon=False, fontsize=8.3, handlelength=2.2)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig2_fullpop_{source}.{ext}")
    plt.close(fig)
print(f"\nwrote {out_md}, {out_json}, figures -> {FIG}/fig2_fullpop_*.{{png,pdf}}")
