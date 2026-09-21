#!/usr/bin/env python3
"""Answer-span confidence statistics vs whole-reply confidence (post-hoc).

Reads the per-token re-collections in data/span_conf/*.jsonl (see
scripts/collect_answer_span_conf.py), extracts the answer span of each STOP
reply, and recomputes HELP-vs-rest ROC for:

    reply_mean   whole-reply mean token logprob  (paper's statistic, sanity)
    span_mean    answer-span mean token logprob  (= log geometric mean prob)
    span_min     answer-span minimum token logprob
    first_tok    first answer-token logprob

Span rule: text after the last `Answer:` marker when present (the prompt asks
for a terminal `Answer: <...>` line), else the whole reply. Whitespace-only
tokens are dropped. Labels come from the ORIGINAL campaign logs (strict
cross-replay four-way; AMBIGUOUS excluded) — this pass never rescores.

Rows whose regenerated text does not byte-match the logged generation are kept
(greedy + fixed seed makes them rare) but counted; the md reports the match
rate per run.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

THR = 0.5
MARK = re.compile(r"Answer\s*:", re.I)
RUNS = [  # (display, source log, span_conf file)
    ("Qwen / vqav2 (lockbox)", "logs/lockbox_vqav2.jsonl", "data/span_conf/lockbox_vqav2.jsonl"),
    ("Qwen / textvqa (lockbox)", "logs/lockbox_textvqa.jsonl", "data/span_conf/lockbox_textvqa.jsonl"),
    ("Qwen one-word / vqav2 (lockbox)", "logs/lockbox_vqav2_qwen_oneword.jsonl", "data/span_conf/lockbox_vqav2_oneword.jsonl"),
    ("FRESH Qwen / vqav2", "logs/fresh_vqav2_orig.jsonl", "data/span_conf/fresh_vqav2_orig.jsonl"),
    ("FRESH Qwen one-word / vqav2", "logs/fresh_vqav2_oneword.jsonl", "data/span_conf/fresh_vqav2_oneword.jsonl"),
    ("FRESH Qwen / textvqa", "logs/fresh_textvqa_orig.jsonl", "data/span_conf/fresh_textvqa_orig.jsonl"),
    ("FRESH Qwen one-word / textvqa", "logs/fresh_textvqa_oneword.jsonl", "data/span_conf/fresh_textvqa_oneword.jsonl"),
]
STATS = ("reply_mean", "span_mean", "span_min", "first_tok")


def four_way(s1, s2, x1, x2):
    b = lambda v: v >= THR
    d1, d2 = int(b(x1)) - int(b(s1)), int(b(x2)) - int(b(s2))
    if d1 > 0 and d2 > 0:
        return "HELP"
    if d1 < 0 and d2 < 0:
        return "HARM"
    if d1 == 0 and d2 == 0:
        return "NEUTRAL"
    return "AMBIGUOUS"


def labels_from_log(log):
    by = defaultdict(dict)
    for line in open(ROOT / log):
        r = json.loads(line)
        if r.get("type") == "rollout":
            by[r["sample_id"]][(r["action"]["type"], r["replay"])] = r
    lab, verbose = {}, {}
    for sid, d in by.items():
        need = [("STOP", 1), ("STOP", 2), ("HIGHRES", 1), ("HIGHRES", 2)]
        if not all(k in d for k in need):
            continue
        sc = {k: float(d[k]["score_after"]) for k in need}
        lab[sid] = four_way(sc[("STOP", 1)], sc[("STOP", 2)],
                            sc[("HIGHRES", 1)], sc[("HIGHRES", 2)])
        verbose[sid] = len((d[("STOP", 1)]["answer_after"] or "").split()) > 1
    return lab, verbose


def span_stats(tokens, lps):
    if not tokens:
        return None
    text = "".join(tokens)
    ends = np.cumsum([len(t) for t in tokens])
    starts = ends - np.array([len(t) for t in tokens])
    matches = list(MARK.finditer(text))
    lo = matches[-1].end() if matches else 0
    idx = [i for i in range(len(tokens))
           if starts[i] >= lo and tokens[i].strip() != ""]
    if not idx:  # marker present but nothing after it: fall back to whole reply
        idx = [i for i in range(len(tokens)) if tokens[i].strip() != ""] or [len(tokens) - 1]
    span = [lps[i] for i in idx]
    return {"reply_mean": float(np.mean(lps)),
            "span_mean": float(np.mean(span)),
            "span_min": float(min(span)),
            "first_tok": float(span[0]),
            "span_len": len(span)}


def roc_ci(y, s, rng, n=3000):
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            v.append(roc_auc_score(y[i], s[i]))
    return [float(q) for q in np.quantile(v, [0.025, 0.975])]


def main():
    rng = np.random.default_rng(20260828)
    out = []
    for disp, log, sc_file in RUNS:
        if not (ROOT / sc_file).exists():
            print("missing", sc_file, "— skip")
            continue
        lab, verbose = labels_from_log(log)
        rows, n_match, n_gen = [], 0, 0
        for line in open(ROOT / sc_file):
            r = json.loads(line)
            if r.get("type") != "span_conf":
                continue
            sid = r["sample_id"]
            if sid not in lab:
                continue
            st = [span_stats(r[f"r{k}"]["tokens"], r[f"r{k}"]["logprobs"]) for k in (1, 2)]
            if any(s is None for s in st):
                continue
            n_gen += 2
            n_match += sum(int(r[f"r{k}"]["matches_logged"]) for k in (1, 2))
            row = {"label": lab[sid], "verbose": verbose[sid]}
            for k in STATS:
                row[k] = float(np.mean([s[k] for s in st]))
            rows.append(row)
        use = [r for r in rows if r["label"] != "AMBIGUOUS"]
        y = np.array([r["label"] == "HELP" for r in use], int)
        if not (0 < y.sum() < len(y)):
            continue
        entry = {"run": disp, "n_eval": len(use), "n_help": int(y.sum()),
                 "labels": dict(Counter(r["label"] for r in rows)),
                 "gen_match_rate": round(n_match / max(n_gen, 1), 4)}
        for stat in STATS:
            s = -np.array([r[stat] for r in use])
            entry[stat] = {"roc": float(roc_auc_score(y, s)), "ci": roc_ci(y, s, rng)}
            ow = [i for i, r in enumerate(use) if not r["verbose"]]
            y1 = y[ow]
            if 0 < y1.sum() < len(y1):
                entry[stat]["roc_oneword_only"] = float(
                    roc_auc_score(y1, -np.array([use[i][stat] for i in ow])))
        out.append(entry)
        print(f"{disp:34s} n={len(use):3d} H={y.sum():3d} match={entry['gen_match_rate']:.3f} "
              + " ".join(f"{k} {entry[k]['roc']:.3f}" for k in STATS))

    Path(ROOT / "paper/answer_span_confidence.json").write_text(json.dumps(out, indent=1))
    lines = [
        "# Answer-span confidence (post-hoc re-collection)",
        "",
        "HELP-vs-rest ROC, strict labels from the original logs (AMBIGUOUS excluded),",
        "confidence statistics recomputed from per-token logprobs of regenerated STOP",
        "replies (greedy, seed 1234, serial; `match` = share of regenerations that",
        "byte-match the logged text). Span = text after the terminal `Answer:` marker",
        "(whole reply when absent).",
        "",
        "| run | n (HELP) | match | reply-mean ROC [CI] | span-mean ROC [CI] | span-min ROC [CI] | first-token ROC [CI] |",
        "|---|--:|--:|:--:|:--:|:--:|:--:|",
    ]
    for e in out:
        def f(k):
            d = e[k]
            return f"{d['roc']:.3f} [{d['ci'][0]:.2f}, {d['ci'][1]:.2f}]"
        lines.append(f"| {e['run']} | {e['n_eval']} ({e['n_help']}) | "
                     f"{100 * e['gen_match_rate']:.0f}% | " +
                     " | ".join(f(k) for k in STATS) + " |")
    lines += ["", "One-word-STOP-only ROCs are in paper/answer_span_confidence.json "
              "(`roc_oneword_only`)."]
    Path(ROOT / "paper/answer_span_confidence.md").write_text("\n".join(lines) + "\n")
    print("wrote paper/answer_span_confidence.{json,md}")


if __name__ == "__main__":
    main()
