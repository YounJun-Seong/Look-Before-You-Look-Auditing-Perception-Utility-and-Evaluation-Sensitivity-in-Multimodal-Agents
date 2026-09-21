"""Gate 1 audit: turn rollouts into the decision table (SPEC §7, §9).

Everything here is computed from the JSONL log alone — no model, no GPU — so
the audit can be re-run whenever a scoring rule changes.

Three conventions that the whole table depends on:

1. **Strict labels.** An action "solves" a sample only if it scores >= 0.5 under
   BOTH replays. R2 permutes MCQ options / paraphrases the question, so a
   letter that was guessed rarely survives. Diversity, exclusive-solve, harm and
   the decisive set are all defined on these strict labels.
2. **Cross-fitted oracle.** The best action is chosen on R1 and scored on R2.
   Choosing and scoring on the same replay pays the winner's curse of a max over
   11 candidates; with 5-way MCQ that alone can manufacture double-digit gains.
3. **Luck floor.** The same statistics are computed over a cardinality-matched
   dummy action set (repeat / random crops / re-encodes). Reported headroom is
   net of that floor, because 11 uninformative draws also raise a max.

Ties are broken by cost (STOP < OCR < CROP < HIGHRES), so "best action" never
depends on argmax scan order.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .schema import COST_RANK

FAMILIES = ("STOP", "CROP", "OCR", "HIGHRES")
THRESHOLD = 0.5


@dataclass
class SampleView:
    """One sample's outcomes, indexed by action key."""

    sample_id: str
    dataset: str
    r1: dict[str, float] = field(default_factory=dict)
    r2: dict[str, float] = field(default_factory=dict)
    family: dict[str, str] = field(default_factory=dict)
    dummy_r1: dict[str, float] = field(default_factory=dict)
    dummy_r2: dict[str, float] = field(default_factory=dict)
    dummy_family: dict[str, str] = field(default_factory=dict)
    full_r1: Optional[float] = None
    full_r2: Optional[float] = None
    parse_fail: int = 0
    n_rows: int = 0

    def strict(self, key: str) -> int:
        return int(self.r1.get(key, 0.0) >= THRESHOLD and self.r2.get(key, 0.0) >= THRESHOLD)

    @property
    def keys(self) -> list[str]:
        return sorted(self.r1.keys() & self.r2.keys())

    def solved_families(self) -> set[str]:
        return {self.family[k] for k in self.keys if self.strict(k)}


def load(records: Iterable[dict]) -> dict[str, SampleView]:
    views: dict[str, SampleView] = {}
    for rec in records:
        sid = rec["sample_id"]
        v = views.setdefault(sid, SampleView(sid, rec["dataset"]))
        v.n_rows += 1
        if not rec.get("parse_ok", True):
            v.parse_fail += 1
        key = rec["action"]["key"]
        fam = rec["action"].get("family") or rec["action"]["type"]
        score = float(rec["score_after"])
        if rec["action"]["type"] == "FULL":
            if rec["replay"] == 1:
                v.full_r1 = score
            else:
                v.full_r2 = score
            continue
        if rec.get("dummy"):
            (v.dummy_r1 if rec["replay"] == 1 else v.dummy_r2)[key] = score
            v.dummy_family[key] = fam
        else:
            (v.r1 if rec["replay"] == 1 else v.r2)[key] = score
            v.family[key] = fam
    return views


def load_path(path: str | Path) -> dict[str, SampleView]:
    recs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "rollout":
                recs.append(obj)
    return load(recs)


# --- selection --------------------------------------------------------------

def _argmax_by_cost(keys: list[str], scores: dict[str, float],
                    fam: dict[str, str]) -> Optional[str]:
    if not keys:
        return None
    return min(keys, key=lambda k: (-scores.get(k, 0.0),
                                    COST_RANK.get(fam.get(k, "CROP"), 9), k))


def oracle_cf(v: SampleView, dummy: bool = False) -> float:
    """Choose on R1 (cost-broken ties), score on R2."""
    r1 = v.dummy_r1 if dummy else v.r1
    r2 = v.dummy_r2 if dummy else v.r2
    fam = v.dummy_family if dummy else v.family
    keys = sorted(r1.keys() & r2.keys())
    best = _argmax_by_cost(keys, r1, fam)
    return r2.get(best, 0.0) if best else 0.0


def oracle_naive(v: SampleView, dummy: bool = False) -> float:
    r1 = v.dummy_r1 if dummy else v.r1
    return max(r1.values()) if r1 else 0.0


def best_action_family(v: SampleView) -> Optional[str]:
    """Best action *type* under strict labels, cheapest-wins ties."""
    solved = [k for k in v.keys if v.strict(k)]
    if not solved:
        return None
    best = min(solved, key=lambda k: (COST_RANK.get(v.family[k], 9), k))
    return v.family[best]


def fixed_policy_keys(views: dict[str, SampleView]) -> list[str]:
    """Action keys usable as an 'always X' policy across all samples.

    CROP boxes differ per sample, so a fixed crop policy is defined by slot
    (always take candidate k), not by coordinates.
    """
    keys = {"STOP#0", "OCR#0", "HIGHRES#0"}
    return sorted(keys) + [f"CROP#{i}" for i in range(8)]


def _policy_score(v: SampleView, policy: str, replay: int) -> float:
    table = v.r1 if replay == 1 else v.r2
    if policy.startswith("CROP#"):
        slot = policy.split("#")[1]
        for k in table:
            if k.startswith(f"CROP#{slot}#"):
                return table[k]
        return 0.0
    return table.get(policy, 0.0)


def best_fixed_cf(views: list[SampleView]) -> tuple[float, str, dict[str, float]]:
    """2-fold cross-fitted best fixed policy (SPEC §7.4).

    Selecting the winning 'always X' on the same data it is scored on would
    hand the fixed-policy baseline its own winner's curse, and the policy
    headroom P is a *difference* of two selections — both sides must be fair.

    Returns the mean, the picks (one per fold), and the per-sample score under
    the pick trained on the OTHER fold, so P has a paired vector for its CI.
    """
    if not views:
        return 0.0, "none", {}
    policies = fixed_policy_keys({v.sample_id: v for v in views})
    order = sorted(views, key=lambda v: v.sample_id)
    # Random repeated 2-fold splits, not an every-other-row split: a
    # deterministic split can align with structure in the sample ids and then
    # systematically trains on the half where the other policy wins.
    rng = np.random.default_rng(20260816)
    n_splits = 10
    acc: dict[str, list[float]] = {v.sample_id: [] for v in order}
    chosen: Counter = Counter()
    for _ in range(n_splits):
        perm = rng.permutation(len(order))
        folds = [[order[i] for i in perm[0::2]], [order[i] for i in perm[1::2]]]
        for i in (0, 1):
            train, test = folds[i], folds[1 - i]
            if not train or not test:
                continue
            pick = max(policies, key=lambda p: (
                float(np.mean([_policy_score(v, p, 1) for v in train])),
                -COST_RANK.get(p.split("#")[0], 9)))
            chosen[pick] += 1
            for v in test:
                acc[v.sample_id].append(_policy_score(v, pick, 2))
    per_sample = {sid: float(np.mean(xs)) for sid, xs in acc.items() if xs}
    mean = float(np.mean(list(per_sample.values()))) if per_sample else 0.0
    label = ",".join(f"{p}:{c}" for p, c in chosen.most_common(3))
    return mean, label, per_sample


# --- audit ------------------------------------------------------------------

def _bootstrap_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 20260816,
                  alpha: float = 0.05) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def audit(views: dict[str, SampleView], label: str = "pooled",
          n_boot: int = 10000) -> dict:
    vs = sorted(views.values(), key=lambda v: v.sample_id)
    n = len(vs)
    if n == 0:
        return {"label": label, "n": 0}

    base_r2 = np.array([v.r2.get("STOP#0", 0.0) for v in vs])
    base_r1 = np.array([v.r1.get("STOP#0", 0.0) for v in vs])
    orc = np.array([oracle_cf(v) for v in vs])
    orc_naive = np.array([oracle_naive(v) for v in vs])
    has_dummy = [v for v in vs if v.dummy_r1 and v.dummy_r2]
    dummy = np.array([oracle_cf(v, dummy=True) for v in has_dummy]) if has_dummy else np.array([])
    full = np.array([v.full_r2 for v in vs if v.full_r2 is not None])

    fixed_acc = {p: float(np.mean([_policy_score(v, p, 2) for v in vs]))
                 for p in fixed_policy_keys(views)}
    bf, bf_choice, bf_per_sample = best_fixed_cf(vs)

    # H: oracle net of the higher of (base, luck floor). Both terms are paired
    # per sample, and when the dummy arm is the binding floor the comparison is
    # restricted to the samples that actually have a dummy arm.
    dummy_mean = float(dummy.mean()) if len(dummy) else None
    if dummy_mean is not None and dummy_mean > float(base_r2.mean()):
        sub = np.array([oracle_cf(v) for v in has_dummy])
        h_vec = sub - dummy
        h_floor = "dummy"
    else:
        h_vec = orc - base_r2
        h_floor = "base"
    h_point = float(h_vec.mean())
    h_lo, h_hi = _bootstrap_ci(h_vec, n_boot)

    p_vec = np.array([oracle_cf(v) - bf_per_sample.get(v.sample_id, 0.0)
                      for v in vs if v.sample_id in bf_per_sample])
    p_point = float(p_vec.mean()) if len(p_vec) else 0.0
    p_lo, p_hi = _bootstrap_ci(p_vec, n_boot)

    # Strict-label structure.
    decisive = [v for v in vs if 0 < sum(v.strict(k) for k in v.keys) < len(v.keys)]
    div = Counter(best_action_family(v) for v in decisive)
    div.pop(None, None)
    div_total = sum(div.values()) or 1

    exclusive = {}
    for fam in FAMILIES:
        cnt = 0
        for v in vs:
            fams = v.solved_families()
            if fams == {fam}:
                cnt += 1
        exclusive[fam] = cnt / n

    # Harm: STOP was right and some action broke it.
    harm_cases = [v for v in vs
                  if v.strict("STOP#0") and any(not v.strict(k) for k in v.keys
                                                if v.family[k] != "STOP")]
    harm_by_family = {}
    for fam in ("CROP", "OCR", "HIGHRES"):
        d = n_h = 0
        for v in vs:
            if not v.strict("STOP#0"):
                continue
            ks = [k for k in v.keys if v.family[k] == fam]
            if not ks:
                continue
            n_h += 1
            d += int(any(not v.strict(k) for k in ks))
        harm_by_family[fam] = d / n_h if n_h else 0.0

    # Recovery overlap among base failures — the Venn table (user, 2026-08-16).
    base_fail = [v for v in vs if not v.strict("STOP#0")]
    overlap: Counter = Counter()
    for v in base_fail:
        fams = tuple(sorted(v.solved_families() - {"STOP"}))
        overlap[fams or ("none",)] += 1

    # Label reliability.
    agree, tot = 0, 0
    for v in vs:
        for k in v.keys:
            tot += 1
            agree += int((v.r1[k] >= THRESHOLD) == (v.r2[k] >= THRESHOLD))
    top1_agree, top1_tot = 0, 0
    for v in decisive:
        f1 = _argmax_by_cost(v.keys, v.r1, v.family)
        f2 = _argmax_by_cost(v.keys, v.r2, v.family)
        if f1 and f2:
            top1_tot += 1
            top1_agree += int(v.family[f1] == v.family[f2])

    return {
        "label": label,
        "n": n,
        "accuracy": {
            "coarse_stop_r2": float(base_r2.mean()),
            "coarse_stop_r1": float(base_r1.mean()),
            "full_image_r2": float(full.mean()) if len(full) else None,
            "oracle_cf": float(orc.mean()),
            "oracle_naive_r1": float(orc_naive.mean()),
            "oracle_dummy_cf": dummy_mean,
            "fixed_policies": fixed_acc,
            "best_fixed_cf": bf,
            "best_fixed_choice": bf_choice,
        },
        "headroom": {
            "H_net": h_point, "H_ci": [h_lo, h_hi], "H_floor": h_floor,
            "P_policy": p_point, "P_ci": [p_lo, p_hi],
            "oracle_minus_base": float(orc.mean() - base_r2.mean()),
            "luck_floor_gain": (dummy_mean - float(base_r2.mean()))
            if dummy_mean is not None else None,
        },
        "diversity": {
            "n_decisive": len(decisive),
            "distribution": {f: div.get(f, 0) / div_total for f in FAMILIES},
            "counts": {f: div.get(f, 0) for f in FAMILIES},
            "max_share": max((div.get(f, 0) / div_total for f in FAMILIES), default=0.0),
            "n_families_over_10pct": sum(
                1 for f in FAMILIES if div.get(f, 0) / div_total >= 0.10),
        },
        "exclusive_solve": exclusive,
        "harm": {
            "dmg_rate": len(harm_cases) / n,
            "by_family": harm_by_family,
        },
        "recovery_overlap": {"+".join(k): c for k, c in sorted(
            overlap.items(), key=lambda kv: -kv[1])},
        "reliability": {
            "per_action_agreement": agree / tot if tot else 0.0,
            "top1_family_agreement": top1_agree / top1_tot if top1_tot else 0.0,
            "parse_failures": sum(v.parse_fail for v in vs),
            "rows": sum(v.n_rows for v in vs),
        },
    }


def bottleneck(views: dict[str, SampleView]) -> dict:
    """The subset where the *budget* was the binding constraint.

    coarse-STOP wrong AND the full-information view right: whatever the reader
    lacked, it was resolution, not capability. Action structure inside this
    subset is far more informative than over the whole dataset, where samples
    the model could never answer at any budget dilute everything.

    Requires FULL rows; returns n=0 when the reference arm was not run.
    """
    subset = [v for v in sorted(views.values(), key=lambda v: v.sample_id)
              if v.full_r1 is not None and v.full_r2 is not None
              and not v.strict("STOP#0")
              and v.full_r1 >= THRESHOLD and v.full_r2 >= THRESHOLD]
    n = len(subset)
    if n == 0:
        return {"n": 0, "note": "no FULL rows, or no coarse-wrong/full-right samples"}

    solve_rate = {}
    for fam in FAMILIES:
        solve_rate[fam] = sum(
            1 for v in subset if any(v.strict(k) for k in v.keys if v.family[k] == fam)
        ) / n
    best = Counter(best_action_family(v) for v in subset)
    best.pop(None, None)
    tot = sum(best.values()) or 1
    exclusive = {}
    for fam in FAMILIES:
        exclusive[fam] = sum(1 for v in subset if v.solved_families() == {fam}) / n
    recovered = sum(1 for v in subset if v.solved_families() - {"STOP"})
    return {
        "n": n,
        "share_of_all": n / len(views),
        "recovery_rate": recovered / n,
        "solve_rate_by_family": solve_rate,
        "best_family": {f: best.get(f, 0) / tot for f in FAMILIES},
        "best_family_counts": {f: best.get(f, 0) for f in FAMILIES},
        "exclusive_solve": exclusive,
        "overlap": {"+".join(k) if k else "none": c for k, c in Counter(
            tuple(sorted(v.solved_families() - {"STOP"})) for v in subset
        ).most_common()},
    }


def verdict(pooled: dict, per_dataset: list[dict]) -> dict:
    """Apply the pre-registered decision table (SPEC §9). No judgement calls at
    analysis time — the thresholds were fixed before the data existed."""
    H = pooled["headroom"]["H_net"]
    H_lo = pooled["headroom"]["H_ci"][0]
    P = pooled["headroom"]["P_policy"]
    P_lo = pooled["headroom"]["P_ci"][0]
    dmg = pooled["harm"]["dmg_rate"]

    def diverse(d: dict) -> bool:
        return (d["diversity"]["max_share"] <= 0.60
                and d["diversity"]["n_families_over_10pct"] >= 3)

    n_div = sum(1 for d in per_dataset if diverse(d))
    if (H >= 0.10 and H_lo > 0.05 and P >= 0.06 and P_lo > 0.02
            and n_div >= 3 and dmg >= 0.08):
        v = "STRONG-GO"
    elif H >= 0.06 and H_lo > 0.02 and P >= 0.04 and P_lo > 0.0 and n_div >= 2:
        v = "GO"
    elif H < 0.04 or (P < 0.02 and n_div == 0):
        v = "STOP"
    else:
        v = "CONDITIONAL"
    return {"verdict": v, "H_net": H, "H_ci_low": H_lo, "P_policy": P,
            "P_ci_low": P_lo, "dmg_rate": dmg, "datasets_diverse": n_div,
            "n_datasets": len(per_dataset)}


def by_dataset(views: dict[str, SampleView], n_boot: int = 2000) -> list[dict]:
    groups: dict[str, dict[str, SampleView]] = defaultdict(dict)
    for sid, v in views.items():
        groups[v.dataset][sid] = v
    return [audit(g, label=name, n_boot=n_boot) for name, g in sorted(groups.items())]


def by_meta(views: dict[str, SampleView], meta: dict[str, dict], key: str,
            n_boot: int = 500, min_n: int = 3) -> list[dict]:
    """Break the audit down by a manifest metadata field (domain, subtask).

    Joined from the manifest rather than stored per rollout: the manifest is a
    committed artifact, and the log stays the record of what was *executed*.
    Small strata are kept — with n=4 the CIs are meaningless, but the raw
    action counts are exactly what a later shortcut analysis needs.
    """
    groups: dict[str, dict[str, SampleView]] = defaultdict(dict)
    for sid, v in views.items():
        groups[str(meta.get(sid, {}).get(key, "_none"))][sid] = v
    out = []
    for name, g in sorted(groups.items()):
        if len(g) < min_n:
            out.append({"label": name, "n": len(g), "note": "too small to audit"})
            continue
        out.append(audit(g, label=name, n_boot=n_boot))
    return out


def render_markdown(pooled: dict, per_dataset: list[dict], verd: dict,
                    bottle: dict | None = None) -> str:
    """The Gate 1 Audit Table (SPEC §13, Table 1-3)."""
    pct = lambda x: "—" if x is None else f"{100 * x:.1f}"
    rows = [
        "# Gate 1 Audit Table",
        "",
        f"**Verdict: {verd['verdict']}** (pre-registered thresholds, SPEC §9)  ",
        f"H_net = {pct(verd['H_net'])}pp (CI low {pct(verd['H_ci_low'])}), "
        f"P_policy = {pct(verd['P_policy'])}pp (CI low {pct(verd['P_ci_low'])}), "
        f"DMG = {pct(verd['dmg_rate'])}%, "
        f"diverse datasets {verd['datasets_diverse']}/{verd['n_datasets']}",
        "",
        "## Table 1 — accuracy and headroom (%)",
        "",
        "| set | n | coarse STOP | full image | best fixed (cf) | naive oracle | "
        "oracle (cf) | dummy oracle | H net | P policy |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for d in [pooled] + per_dataset:
        a, h = d["accuracy"], d["headroom"]
        rows.append(
            f"| {d['label']} | {d['n']} | {pct(a['coarse_stop_r2'])} | "
            f"{pct(a['full_image_r2'])} | {pct(a['best_fixed_cf'])} | "
            f"{pct(a['oracle_naive_r1'])} | "
            f"{pct(a['oracle_cf'])} | {pct(a['oracle_dummy_cf'])} | "
            f"{pct(h['H_net'])} | {pct(h['P_policy'])} |")

    rows += ["", "## Table 2 — best action type on decisive samples (%)", "",
             "| set | n decisive | STOP | CROP | OCR | HIGHRES | max share |",
             "|---|--:|--:|--:|--:|--:|--:|"]
    for d in [pooled] + per_dataset:
        dv = d["diversity"]
        rows.append(
            f"| {d['label']} | {dv['n_decisive']} | "
            + " | ".join(pct(dv["distribution"][f]) for f in FAMILIES)
            + f" | {pct(dv['max_share'])} |")

    rows += ["", "## Table 3 — exclusive solve and harm (%)", "",
             "| set | excl STOP | excl CROP | excl OCR | excl HIGHRES | DMG | "
             "harm CROP | harm OCR | harm HIGHRES |", "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for d in [pooled] + per_dataset:
        ex, hm = d["exclusive_solve"], d["harm"]
        rows.append(
            f"| {d['label']} | " + " | ".join(pct(ex[f]) for f in FAMILIES)
            + f" | {pct(hm['dmg_rate'])} | "
            + " | ".join(pct(hm["by_family"][f]) for f in ("CROP", "OCR", "HIGHRES"))
            + " |")

    rows += ["", "## Table 4 — recovery overlap among base failures (counts)", "",
             "| recovering action types | samples |", "|---|--:|"]
    for k, c in pooled["recovery_overlap"].items():
        rows.append(f"| {k} | {c} |")

    if bottle and bottle.get("n"):
        rows += ["", "## Table 5 — budget-bottleneck subset "
                 "(coarse wrong AND full-image right)", "",
                 f"n = {bottle['n']} ({pct(bottle['share_of_all'])}% of samples); "
                 f"recovered by at least one action: {pct(bottle['recovery_rate'])}%",
                 "",
                 "| | STOP | CROP | OCR | HIGHRES |", "|---|--:|--:|--:|--:|",
                 "| solves the sample | "
                 + " | ".join(pct(bottle["solve_rate_by_family"][f]) for f in FAMILIES) + " |",
                 "| is the best action | "
                 + " | ".join(pct(bottle["best_family"][f]) for f in FAMILIES) + " |",
                 "| solves it exclusively | "
                 + " | ".join(pct(bottle["exclusive_solve"][f]) for f in FAMILIES) + " |",
                 "", "recovery overlap inside the subset: "
                 + ", ".join(f"{k}={v}" for k, v in bottle["overlap"].items())]

    rel = pooled["reliability"]
    rows += ["", "## Table 6 — label reliability", "",
             f"- per-action R1/R2 agreement: {pct(rel['per_action_agreement'])}%",
             f"- top-1 family agreement (decisive): {pct(rel['top1_family_agreement'])}%",
             f"- parse failures: {rel['parse_failures']} / {rel['rows']} rows",
             ""]
    return "\n".join(rows)
