#!/usr/bin/env python3
"""Post-hoc deployable multi-action selector on the Gate 0 brute-force logs.

Reviewer gap this closes: the paper audits a 4-family action space but the
deployable results reduce to Stop-vs-HighRes, and the headline +9.8pp is a
cross-replay oracle, not a policy anyone can run. Here we train, on the same
16 pre-action features the frozen gate uses, per-family value heads
(dScore(a) = score(a) - score(STOP)) and act with

    pi_lambda(s) = argmax_a  dScore_hat(a|s) - lambda * cost(a|s),   STOP = 0

sweeping lambda to trace an accuracy/cost Pareto front. Costs stay on two
separate axes (visual tokens; added wall-clock latency) because OCR is free in
tokens and expensive in seconds -- summing them is forbidden by the audit
design.

Deployability rules:
  * CROP executes a uniformly chosen grid crop -> evaluated as the mean score
    over the state's grid-crop arms (expectation, no within-family peeking).
    A diagnostic row with the within-family best grid crop is reported
    separately as a non-deployable bound.
  * Features are pre-action only (question, provisional STOP answer, STOP
    confidence). conf_rank is normalised with the training fold's empirical
    CDF -- never with test statistics.
  * All learned rows are out-of-fold (5-fold CV, stratified by dataset x
    any-family-HELP); best-fixed is cross-fitted inside the same folds.
  * The cross-replay oracle (select on R1, score on R2, symmetrised) is the
    only row allowed to peek at outcomes; it is the upper bound, not a policy.

Everything is offline re-analysis of logs/g0_mme200.jsonl + logs/tv_200.jsonl
joined with data/stage_a_features.jsonl (STOP confidence). No new rollouts.
Post-hoc: conducted after the confirmatory (lockbox) analyses; labeled as such.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ppu.gate_features import FEATURE_NAMES, extract  # noqa: E402

from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SEED = 20260828
N_FOLDS = 5
N_BOOT = 10_000
THR = 0.5
FAMILIES = ("CROP", "OCR", "HIGHRES")
LOGS = [("logs/g0_mme200.jsonl", "mme"), ("logs/tv_200.jsonl", "textvqa")]
FEATURES_PATH = "data/stage_a_features.jsonl"
OUT_JSON = "paper/multiaction_selector.json"
OUT_MD = "paper/multiaction_selector.md"

LAM_TOK = [0.0] + list(np.logspace(-6, -2, 33))
LAM_LAT = [0.0] + list(np.logspace(-5, -0.5, 33))


# --- data ---------------------------------------------------------------

def load_states():
    """One row per state: per-arm scores for both replays + per-arm costs."""
    by = {}
    for log, ds in LOGS:
        for line in open(ROOT / log):
            r = json.loads(line)
            if r.get("type") != "rollout" or r.get("dummy"):
                continue
            sid = r["sample_id"]
            st = by.setdefault(sid, {
                "sid": sid, "dataset": ds, "question": r["question"],
                "score": defaultdict(dict), "vtok": defaultdict(list),
                "lat": defaultdict(list), "stop_answer": None,
            })
            a, rep = r["action"], r["replay"]
            fam, key = a["family"], a["key"]
            if fam == "CROP" and a.get("source") != "grid":
                continue  # VLM proposals: 70% parse failure in Gate 0, uneven coverage
            st["score"][key][rep] = float(r["score_after"])
            st["vtok"][key].append(float(r["cost"]["visual_tokens"]))
            st["lat"][key].append(float(r["cost"]["latency_s"])
                                  + float(r["cost"].get("ocr_latency_s", 0.0)))
            if fam == "STOP" and rep == 1:
                st["stop_answer"] = r["answer_after"]

    conf = {}
    for line in open(ROOT / FEATURES_PATH):
        r = json.loads(line)
        lps = [r[k]["mean_logprob"] for k in ("stop_r1", "stop_r2")
               if r.get(k) and r[k].get("mean_logprob") is not None]
        if lps:
            conf[r["state_id"]] = {
                "mean_logprob": float(np.mean(lps)),
                "n_tokens": int(r["stop_r1"].get("n_tokens") or 0),
            }

    states, dropped = [], 0
    for sid, st in by.items():
        keys = {k: v for k, v in st["score"].items() if set(v) >= {1, 2}}
        grid = sorted(k for k in keys if k.startswith("CROP"))
        need = {"STOP#0", "OCR#0"} | {k for k in keys if k.startswith("HIGHRES")}
        hi = next((k for k in keys if k.startswith("HIGHRES")), None)
        if sid not in conf or "STOP#0" not in keys or "OCR#0" not in keys \
                or hi is None or not grid:
            dropped += 1
            continue
        full = next((k for k in st["score"] if k.startswith("FULL")
                     and set(st["score"][k]) >= {1, 2}), None)
        s = st["score"]

        def sc(key):  # (r1, r2)
            return np.array([s[key][1], s[key][2]])

        row = {
            "sid": sid, "dataset": st["dataset"], "question": st["question"],
            "answer": st["stop_answer"] or "",
            "mean_logprob": conf[sid]["mean_logprob"],
            "answer_len_tokens": conf[sid]["n_tokens"],
            "stop": sc("STOP#0"), "ocr": sc("OCR#0"), "hi": sc(hi),
            "grid": np.stack([sc(k) for k in grid]),        # (n_grid, 2)
            "full": sc(full) if full else None,
            "cost_vtok": {
                "STOP": 0.0, "OCR": float(np.mean(st["vtok"]["OCR#0"])),
                "HIGHRES": float(np.mean(st["vtok"][hi])),
                "CROP": float(np.mean([np.mean(st["vtok"][k]) for k in grid])),
            },
            "cost_lat": {
                "STOP": 0.0, "OCR": float(np.mean(st["lat"]["OCR#0"])),
                "HIGHRES": float(np.mean(st["lat"][hi])),
                "CROP": float(np.mean([np.mean(st["lat"][k]) for k in grid])),
            },
            "oracle_keys": ["STOP#0", "OCR#0", hi] + grid,
        }
        row["crop"] = row["grid"].mean(axis=0)              # uniform-grid expectation
        row["crop_best"] = None                              # filled by oracle util
        states.append(row)
    return states, dropped


# --- policy evaluation ----------------------------------------------------

def evaluate(states, choice, crop_mode="uniform"):
    """choice: dict sid -> family. Returns accuracy, costs, action mix."""
    scores, vtoks, lats, mix = [], [], [], defaultdict(int)
    for s in states:
        fam = choice[s["sid"]]
        mix[fam] += 1
        if fam == "CROP":
            v = s["grid"].max(axis=0) if crop_mode == "best" else s["crop"]
        else:
            v = s[{"STOP": "stop", "OCR": "ocr", "HIGHRES": "hi"}[fam]]
        scores.append(np.mean(v))
        vtoks.append(s["cost_vtok"][fam])
        lats.append(s["cost_lat"][fam])
    n = len(states)
    return {
        "acc": float(np.mean(scores)),
        "vtok": float(np.mean(vtoks)),
        "lat": float(np.mean(lats)),
        "look_rate": 1.0 - mix["STOP"] / n,
        "mix": {k: round(v / n, 4) for k, v in sorted(mix.items())},
        "_scores": np.array(scores),
    }


def oracle_row(states):
    """Symmetrised cross-replay oracle over the deployable arm set."""
    scores, vtoks, lats, mix = [], [], [], defaultdict(int)
    for s in states:
        arms = [("STOP", s["stop"], 0.0, 0.0), ("OCR", s["ocr"], s["cost_vtok"]["OCR"], s["cost_lat"]["OCR"]),
                ("HIGHRES", s["hi"], s["cost_vtok"]["HIGHRES"], s["cost_lat"]["HIGHRES"])]
        arms += [("CROP", s["grid"][i], s["cost_vtok"]["CROP"], s["cost_lat"]["CROP"])
                 for i in range(len(s["grid"]))]
        picked = []
        for sel, ev in ((0, 1), (1, 0)):
            i = int(np.argmax([a[1][sel] for a in arms]))
            picked.append((arms[i][1][ev], arms[i][2], arms[i][3], arms[i][0]))
        scores.append(np.mean([p[0] for p in picked]))
        vtoks.append(np.mean([p[1] for p in picked]))
        lats.append(np.mean([p[2] for p in picked]))
        for p in picked:
            mix[p[3]] += 0.5
    n = len(states)
    return {"acc": float(np.mean(scores)), "vtok": float(np.mean(vtoks)),
            "lat": float(np.mean(lats)),
            "look_rate": 1.0 - mix["STOP"] / n,
            "mix": {k: round(v / n, 4) for k, v in sorted(mix.items())},
            "_scores": np.array(scores)}


def boot_ci(delta, seed=SEED, n_boot=N_BOOT):
    rng = np.random.default_rng(seed)
    n = len(delta)
    idx = rng.integers(0, n, (n_boot, n))
    means = delta[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# --- learning -------------------------------------------------------------

def conf_rank_fit(train_lp):
    srt = np.sort(train_lp)

    def rank(lp):  # 1 = least confident (lowest mean logprob)
        return 1.0 - np.searchsorted(srt, lp, side="right") / len(srt)
    return rank


def feats(states, rank_fn):
    X = []
    for s in states:
        cr = float(rank_fn(s["mean_logprob"]))
        X.append(extract({"answer": s["answer"], "question": s["question"],
                          "answer_len_tokens": s["answer_len_tokens"]}, cr))
    return np.array(X, float)


def folds(states, n=N_FOLDS, seed=SEED):
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for i, s in enumerate(states):
        help_any = any(np.all(s[k] > s["stop"]) for k in ("crop", "ocr", "hi"))
        strata[(s["dataset"], help_any)].append(i)
    assign = np.zeros(len(states), int)
    for _, idx in sorted(strata.items()):
        idx = np.array(idx)
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            assign[i] = j % n
    return assign


def fit_predict(states):
    """Out-of-fold: per-family dScore ridge, binary HELP-HIGHRES logistic,
    conf_rank, and the fold-cross-fitted best fixed arm."""
    assign = folds(states)
    n = len(states)
    dhat = {f: np.zeros(n) for f in FAMILIES}
    p_bin = np.zeros(n)
    conf = np.zeros(n)
    fixed = [None] * n
    for k in range(N_FOLDS):
        tr = np.where(assign != k)[0]
        te = np.where(assign == k)[0]
        tr_states = [states[i] for i in tr]
        rank_fn = conf_rank_fit(np.array([s["mean_logprob"] for s in tr_states]))
        Xtr, Xte = feats(tr_states, rank_fn), feats([states[i] for i in te], rank_fn)
        scaler = StandardScaler().fit(Xtr)
        Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
        for f in FAMILIES:
            key = {"CROP": "crop", "OCR": "ocr", "HIGHRES": "hi"}[f]
            y = np.array([np.mean(s[key] - s["stop"]) for s in tr_states])
            m = Ridge(alpha=1.0, random_state=0).fit(Xtr, y)
            dhat[f][te] = m.predict(Xte)
        y_help = np.array([int(np.all(s["hi"] > s["stop"])) for s in tr_states])
        lg = LogisticRegression(class_weight="balanced", C=0.5,
                                solver="lbfgs", max_iter=2000, random_state=0)
        lg.fit(Xtr, y_help)
        p_bin[te] = lg.predict_proba(Xte)[:, 1]
        conf[te] = [rank_fn(states[i]["mean_logprob"]) for i in te]
        # cross-fitted best fixed arm (same fixed action for every state)
        cand = {"STOP": lambda s: s["stop"], "OCR": lambda s: s["ocr"],
                "HIGHRES": lambda s: s["hi"], "CROP": lambda s: s["crop"]}
        best = max(cand, key=lambda f: np.mean([np.mean(cand[f](s)) for s in tr_states]))
        for i in te:
            fixed[i] = best
    return dhat, p_bin, conf, fixed


def fit_transfer(train_states, test_states):
    """Train every learned component on one dataset, predict on the other."""
    rank_fn = conf_rank_fit(np.array([s["mean_logprob"] for s in train_states]))
    Xtr, Xte = feats(train_states, rank_fn), feats(test_states, rank_fn)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    dhat = {}
    for f in FAMILIES:
        key = {"CROP": "crop", "OCR": "ocr", "HIGHRES": "hi"}[f]
        y = np.array([np.mean(s[key] - s["stop"]) for s in train_states])
        dhat[f] = Ridge(alpha=1.0, random_state=0).fit(Xtr, y).predict(Xte)
    y_help = np.array([int(np.all(s["hi"] > s["stop"])) for s in train_states])
    lg = LogisticRegression(class_weight="balanced", C=0.5, solver="lbfgs",
                            max_iter=2000, random_state=0).fit(Xtr, y_help)
    p_bin = lg.predict_proba(Xte)[:, 1]
    conf = np.array([rank_fn(s["mean_logprob"]) for s in test_states])
    return dhat, p_bin, conf


def multi_choice(states, dhat, lam, axis):
    choice = {}
    for i, s in enumerate(states):
        vals = {"STOP": 0.0}
        for f in FAMILIES:
            c = s["cost_vtok" if axis == "vtok" else "cost_lat"][f]
            vals[f] = dhat[f][i] - lam * c
        choice[s["sid"]] = max(vals, key=vals.get)
    return choice


def threshold_sweep(states, score, look_fam="HIGHRES"):
    """Look (fixed family) when score >= tau; sweep tau over quantiles."""
    out = []
    qs = np.unique(np.quantile(score, np.linspace(0, 1, 41)))
    for tau in list(qs) + [np.inf]:
        choice = {s["sid"]: (look_fam if score[i] >= tau else "STOP")
                  for i, s in enumerate(states)}
        ev = evaluate(states, choice)
        ev["tau"] = float(tau)
        out.append(ev)
    return out


def budget_table(ss, sweeps, dhat_loc, conf_loc, p_bin_loc, hi_vtok):
    """Accuracy at pre-declared token budgets (fractions of always-HIGHRES cost).
    Read-off rule, same for every policy: best-accuracy sweep point with mean
    extra tokens <= budget. Paired bootstrap CIs for multi-action deltas."""
    def rebuild(policy, point):
        if policy == "multi":
            return multi_choice(ss, dhat_loc, point["lambda"], "vtok")
        score = conf_loc if policy == "conf" else p_bin_loc
        return {s["sid"]: ("HIGHRES" if score[i] >= point["tau"] else "STOP")
                for i, s in enumerate(ss)}

    rows = []
    for frac in (0.25, 0.5, 1.0):
        B = hi_vtok * frac
        entry = {"budget_frac_of_highres": frac, "budget_vtok": round(B, 1)}
        chosen = {}
        for policy, sw in (("conf", sweeps["confidence_stop_highres"]),
                           ("binary", sweeps["learned_binary_stop_highres"]),
                           ("multi", sweeps["multiaction_vtok"])):
            ok = [p for p in sw if p["vtok"] <= B + 1e-9]
            best = max(ok, key=lambda p: p["acc"])
            ev = evaluate(ss, rebuild(policy, best))
            chosen[policy] = ev
            entry[policy] = {"acc": ev["acc"], "vtok": ev["vtok"], "lat": ev["lat"],
                             "look_rate": ev["look_rate"], "mix": ev["mix"]}
        for other in ("binary", "conf"):
            d = chosen["multi"]["_scores"] - chosen[other]["_scores"]
            entry[f"multi_minus_{other}"] = {"delta": float(d.mean()),
                                             "ci95": boot_ci(d)}
        rows.append(entry)
    return rows


# --- main -----------------------------------------------------------------

def main():
    states, dropped = load_states()
    print(f"states: {len(states)} (dropped {dropped})",
          {ds: sum(s['dataset'] == ds for s in states) for ds in ('mme', 'textvqa')})
    dhat, p_bin, conf, fixed = fit_predict(states)

    idx_of = {s["sid"]: i for i, s in enumerate(states)}
    subsets = {"pooled": states,
               "mme": [s for s in states if s["dataset"] == "mme"],
               "textvqa": [s for s in states if s["dataset"] == "textvqa"]}

    report = {"n_states": len(states), "dropped": dropped, "seed": SEED,
              "provenance": "post-hoc; Gate 0 logs re-analysis; no new rollouts",
              "conventions": {
                  "accuracy": "mean over R1,R2 of the chosen arm's score",
                  "crop_execution": "uniform grid crop (expectation over grid arms)",
                  "oracle": "cross-replay, symmetrised (R1->R2 and R2->R1)",
                  "costs": "visual tokens and added latency reported separately"},
              "subsets": {}}

    for name, ss in subsets.items():
        loc = [idx_of[s["sid"]] for s in ss]
        rows = {}
        for fam in ("STOP", "HIGHRES", "OCR", "CROP"):
            rows[f"always_{fam}"] = evaluate(ss, {s["sid"]: fam for s in ss})
        rows["best_fixed_cf"] = evaluate(ss, {s["sid"]: fixed[idx_of[s["sid"]]] for s in ss})
        rows["oracle_cross_replay"] = oracle_row(ss)
        full = [np.mean(s["full"]) for s in ss if s["full"] is not None]
        rows["full_reference"] = {"acc": float(np.mean(full)), "n": len(full)}

        sweeps = {
            "confidence_stop_highres": threshold_sweep(ss, conf[loc]),
            "learned_binary_stop_highres": threshold_sweep(ss, p_bin[loc]),
        }
        for axis, lams in (("vtok", LAM_TOK), ("lat", LAM_LAT)):
            pts = []
            for lam in lams:
                ch = multi_choice(ss, {f: dhat[f][loc] for f in FAMILIES}, lam, axis)
                ev = evaluate(ss, ch)
                ev["lambda"] = float(lam)
                pts.append(ev)
            sweeps[f"multiaction_{axis}"] = pts
        # non-deployable diagnostic: same selector, within-family best grid crop
        ch0 = multi_choice(ss, {f: dhat[f][loc] for f in FAMILIES}, 0.0, "vtok")
        rows["multiaction_lam0"] = sweeps["multiaction_vtok"][0]
        rows["multiaction_lam0_bestcrop_diag"] = evaluate(ss, ch0, crop_mode="best")

        btable = budget_table(ss, sweeps, {f: dhat[f][loc] for f in FAMILIES},
                              conf[loc], p_bin[loc], rows["always_HIGHRES"]["vtok"])

        stop_scores = evaluate(ss, {s["sid"]: "STOP" for s in ss})["_scores"]
        for key in ("multiaction_lam0", "best_fixed_cf", "oracle_cross_replay"):
            rows[key]["delta_vs_stop"] = float(rows[key]["_scores"].mean() - stop_scores.mean())
            rows[key]["delta_ci95"] = boot_ci(rows[key]["_scores"] - stop_scores)

        for r in rows.values():
            r.pop("_scores", None)
        for sw in sweeps.values():
            for p in sw:
                p.pop("_scores", None)
        report["subsets"][name] = {"rows": rows, "sweeps": sweeps,
                                   "budget_table": btable}

    # cross-dataset transfer: every learned piece fitted on the other dataset
    report["transfer"] = {}
    for src, dst in (("mme", "textvqa"), ("textvqa", "mme")):
        tr, te = subsets[src], subsets[dst]
        dhat_t, p_bin_t, conf_t = fit_transfer(tr, te)
        entry = {"sweeps": {
            "confidence_stop_highres": threshold_sweep(te, conf_t),
            "learned_binary_stop_highres": threshold_sweep(te, p_bin_t),
        }}
        for axis, lams in (("vtok", LAM_TOK), ("lat", LAM_LAT)):
            pts = []
            for lam in lams:
                ev = evaluate(te, multi_choice(te, dhat_t, lam, axis))
                ev["lambda"] = float(lam)
                pts.append(ev)
            entry["sweeps"][f"multiaction_{axis}"] = pts
        for sw in entry["sweeps"].values():
            for p in sw:
                p.pop("_scores", None)
        report["transfer"][f"{src}->{dst}"] = entry

    Path(ROOT / OUT_JSON).write_text(json.dumps(report, indent=1))
    print("wrote", OUT_JSON)
    return report


if __name__ == "__main__":
    main()
