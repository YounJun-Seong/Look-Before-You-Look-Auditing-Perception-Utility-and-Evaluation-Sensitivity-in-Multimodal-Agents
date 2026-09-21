#!/usr/bin/env python
"""Score the pre-registered predictions P1–P5 (paper/PREREG_FRESH_LOCKBOX.md)
from paper/format_confound.json. Run scripts/format_confound_check.py first."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / "paper/format_confound.json"))
F1, F2 = R["FRESH Qwen original / vqav2"], R["FRESH Qwen one-word / vqav2"]
F3, F4 = R["FRESH Qwen original / textvqa"], R["FRESH Qwen one-word / textvqa"]
s, l = "strict", "lenient"
checks = [
 ("P1", "F1 strict: ROC<0.55, verbose>25%, format-flip HELP>=50%",
  F1[s]["conf_roc"] < 0.55 and F1[s]["verbose_stop_share"] > 0.25 and (F1[s]["help_format_flip_share"] or 0) >= 0.5,
  f"ROC {F1[s]['conf_roc']:.3f}, verbose {100*F1[s]['verbose_stop_share']:.0f}%, flips {100*(F1[s]['help_format_flip_share'] or 0):.0f}%"),
 ("P2", "F1 lenient ROC >= F1 strict ROC + 0.15",
  F1[l]["conf_roc"] - F1[s]["conf_roc"] >= 0.15,
  f"{F1[s]['conf_roc']:.3f} -> {F1[l]['conf_roc']:.3f} (Δ {F1[l]['conf_roc']-F1[s]['conf_roc']:+.3f})"),
 ("P3", "F2 strict: verbose<10%, STOP >= F1 STOP+20pp, ROC>0.60, CI lo>0.5",
  F2[s]["verbose_stop_share"] < 0.10 and F2[s]["stop_acc"] >= F1[s]["stop_acc"] + 0.20
  and F2[s]["conf_roc"] > 0.60 and F2[s]["conf_roc_ci"][0] > 0.5,
  f"verbose {100*F2[s]['verbose_stop_share']:.0f}%, STOP {100*F1[s]['stop_acc']:.1f}->{100*F2[s]['stop_acc']:.1f}, ROC {F2[s]['conf_roc']:.3f} [{F2[s]['conf_roc_ci'][0]:.2f}, {F2[s]['conf_roc_ci'][1]:.2f}]"),
 ("P4", "F3, F4 strict ROC > 0.70 and |F3-F4| < 0.10",
  F3[s]["conf_roc"] > 0.70 and F4[s]["conf_roc"] > 0.70 and abs(F3[s]["conf_roc"] - F4[s]["conf_roc"]) < 0.10,
  f"F3 {F3[s]['conf_roc']:.3f}, F4 {F4[s]['conf_roc']:.3f}"),
 ("P5", "lenient: |F1 - F2| < 0.15",
  abs(F1[l]["conf_roc"] - F2[l]["conf_roc"]) < 0.15,
  f"F1 {F1[l]['conf_roc']:.3f}, F2 {F2[l]['conf_roc']:.3f}"),
]
L = ["# Fresh prospective lockbox — pre-registered verdict", "",
     "| | prediction | result | evidence |", "|---|---|:--:|---|"]
for pid, txt, ok, ev in checks:
    L.append(f"| {pid} | {txt} | {'**PASS**' if ok else '**FAIL**'} | {ev} |")
L += ["", "## Full table (strict / lenient)", "",
      "| arm | scorer | STOP | HIGHRES | verbose | HELP (prev.) | flips | conf ROC [CI] | lift |", "|---|---|--:|--:|--:|--:|--:|:--:|--:|"]
for name, E in (("F1 VQAv2 original", F1), ("F2 VQAv2 one-word", F2), ("F3 TextVQA original", F3), ("F4 TextVQA one-word", F4)):
    for sc in (s, l):
        e = E[sc]
        L.append(f"| {name} | {sc} | {100*e['stop_acc']:.1f} | {100*e['highres_acc']:.1f} | {100*e['verbose_stop_share']:.0f}% | "
                 f"{e['labels'].get('HELP',0)} ({100*e['prevalence']:.1f}%) | {100*(e['help_format_flip_share'] or 0):.0f}% | "
                 f"{e['conf_roc']:.3f} [{e['conf_roc_ci'][0]:.2f}, {e['conf_roc_ci'][1]:.2f}] | {e['conf_lift']:.2f}× |")
open(ROOT / "paper/fresh_lockbox_result.md", "w").write("\n".join(L))
print("\n".join(L))
