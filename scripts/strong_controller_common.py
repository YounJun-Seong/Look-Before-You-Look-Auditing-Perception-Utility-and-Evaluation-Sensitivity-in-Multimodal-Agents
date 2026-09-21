"""Shared loaders for the stronger-controller experiment (2026-09-20).

Pooled TextVQA training states with the identical deterministic arm set
(STOP, 8 grid/fallback CROPs, OCR, HIGHRES-512), both replays, STOP logprobs:

  gate0        logs/tv_200.jsonl (+ data/stage_a_features.jsonl for STOP conf)   200
  heldout      logs/fresh_textvqa_orig.jsonl (STOP/HIGHRES) + heldout_textvqa_arms 250
  prospective  logs/prospective_textvqa.jsonl                                      600
  v3           logs/v3_prospective.jsonl                                           300

Every state row has the same structure as scripts/prospective_eval.py uses,
so the frozen v2 selector, the frozen mixture and the Lagrangian oracle can be
evaluated on any of these sets with the same code.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ppu.evaluator import normalize_answer  # noqa: E402
from ppu.gate_features import extract  # noqa: E402

FAMILIES = ("CROP", "OCR", "HIGHRES")
BUDGET_FRACS = (0.25, 0.5, 1.0)
LAM_TOK = [0.0] + list(np.logspace(-6, -2, 33))

POOL = {
    "gate0": ["logs/tv_200.jsonl"],
    "heldout": ["logs/fresh_textvqa_orig.jsonl", "logs/heldout_textvqa_arms.jsonl"],
    "prospective": ["logs/prospective_textvqa.jsonl"],
    "v3": ["logs/v3_prospective.jsonl"],
}


def lenient(pred, golds):  # frozen rule of scripts/p4_lenient_rescore.py
    p = normalize_answer(pred or "")
    toks = p.split()

    def hit(g):
        g = normalize_answer(g)
        return g != "" and (g == p or (toks and g == toks[0]) or f" {g} " in f" {p} ")
    return min(1.0, sum(hit(g) for g in golds) / 3)


def box_key(box):
    return ",".join(f"{v:.4f}" for v in box)


def _stage_a_conf():
    conf = {}
    for line in open(ROOT / "data/stage_a_features.jsonl"):
        r = json.loads(line)
        lps = [r[k]["mean_logprob"] for k in ("stop_r1", "stop_r2")
               if r.get(k) and r[k].get("mean_logprob") is not None]
        if lps:
            conf[r["state_id"]] = {i + 1: lp for i, lp in enumerate(lps)}
    return conf


def load_states(logs, source=None, replays=(1, 2), require_conf=True):
    """Unified state rows. Deterministic arms only (grid CROPs by box key,
    OCR, HIGHRES, STOP). Dummies / VLM-proposed crops / FULL are skipped."""
    by = defaultdict(lambda: {"arms": defaultdict(dict), "ans": defaultdict(dict),
                              "vtok": {}, "conf": {}, "gold": None,
                              "question": None, "answer": None, "alen": 0})
    for log in logs:
        for line in open(ROOT / log):
            r = json.loads(line)
            if r.get("type") != "rollout" or r.get("dummy") or r.get("error"):
                continue
            a, rep = r["action"], r["replay"]
            if rep not in replays:
                continue
            if a["type"] == "STOP":
                key = "STOP"
            elif a["type"] in ("OCR", "HIGHRES"):
                key = a["type"]
            elif a["type"] == "CROP" and a.get("source") == "grid":
                key = f"CROP@{box_key(a['box'])}"
            else:
                continue
            st = by[r["sample_id"]]
            st["arms"][key][rep] = float(r["score_after"])
            st["ans"][key][rep] = r["answer_after"] or ""
            st["vtok"][key] = float(r["cost"]["visual_tokens"])
            st["gold"] = r["gold"]
            st["question"] = r["question"]
            if a["type"] == "STOP":
                c = (r.get("confidence") or {}).get("mean_logprob")
                if c is not None:
                    st["conf"][rep] = c
                if rep == 1:
                    st["answer"] = r["answer_after"] or ""
                    st["alen"] = int(r["cost"].get("tokens_out") or 0)
    if source == "gate0":
        sa = _stage_a_conf()
        for sid, st in by.items():
            if sid in sa:
                st["conf"].update(sa[sid])
    states, dropped = [], 0
    for sid, st in sorted(by.items()):
        keys = sorted(st["arms"])
        complete = (all(set(st["arms"][k]) >= set(replays) for k in keys)
                    and {"STOP", "OCR", "HIGHRES"} <= set(keys)
                    and any(k.startswith("CROP@") for k in keys)
                    and (not require_conf or all(r in st["conf"] for r in (1, 2))))
        if not complete:
            dropped += 1
            continue
        st.update(sid=sid, source=source, keys=keys,
                  crops=[k for k in keys if k.startswith("CROP@")],
                  mean_logprob=(float(np.mean([st["conf"][1], st["conf"][2]]))
                                if all(r in st["conf"] for r in (1, 2)) else None))
        states.append(st)
    return states, dropped


def precompute(states, replays=(1, 2)):
    for st in states:
        st["S"] = {"strict": {}, "lenient": {}}
        for k in st["keys"]:
            st["S"]["strict"][k] = np.array([st["arms"][k][r] for r in replays])
            st["S"]["lenient"][k] = np.array(
                [lenient(st["ans"][k][r], st["gold"]) for r in replays])


def scores(st, key, scorer):
    return st["S"][scorer][key]


def fam_score(st, fam, scorer):
    if fam == "STOP":
        return scores(st, "STOP", scorer)
    if fam == "CROP":  # deployable uniform-grid expectation
        return np.mean([scores(st, k, scorer) for k in st["crops"]], axis=0)
    return scores(st, fam, scorer)


def fam_cost(st, fam):
    if fam == "STOP":
        return 0.0
    if fam == "CROP":
        return float(np.mean([st["vtok"][k] for k in st["crops"]]))
    return st["vtok"][fam]


def load_pool():
    pool = []
    for src, logs in POOL.items():
        ss, dropped = load_states(logs, source=src)
        print(f"  pool[{src}] {len(ss)} states (dropped {dropped})")
        pool += ss
    assert len({s["sid"] for s in pool}) == len(pool), "duplicate state ids across sources"
    precompute(pool)
    return pool


def make_rank(sorted_vals):
    s = np.asarray(sorted_vals, float)
    return lambda v: 1.0 - np.searchsorted(s, v, side="right") / len(s)


def features(states, rank_fn):
    return np.array([extract({"answer": st["answer"], "question": st["question"],
                              "answer_len_tokens": st["alen"]},
                             float(rank_fn(st["mean_logprob"]))) for st in states], float)


def targets(states, scorer="strict"):
    return {f: np.array([float(np.mean(fam_score(st, f, scorer) - fam_score(st, "STOP", scorer)))
                         for st in states]) for f in FAMILIES}


def policy_from_dhat(states, dhat, lam):
    """argmax_f [dhat_f - lam * cost_f], STOP = 0. Returns list of families."""
    out = []
    for i, st in enumerate(states):
        vals = {"STOP": 0.0}
        for f in FAMILIES:
            vals[f] = dhat[f][i] - lam * fam_cost(st, f)
        out.append(max(vals, key=vals.get))
    return out


def policy_value(states, fams, scorer="strict"):
    acc = np.array([float(np.mean(fam_score(st, f, scorer))) for st, f in zip(states, fams)])
    cost = np.array([fam_cost(st, f) for st, f in zip(states, fams)])
    return acc, cost
