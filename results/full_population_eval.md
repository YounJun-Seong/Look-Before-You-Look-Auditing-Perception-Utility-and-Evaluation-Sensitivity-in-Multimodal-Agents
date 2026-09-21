# Full-population deployment evaluation (AMBIGUOUS included)

Frozen gate v2.2 and the confidence selector run over every lockbox state; replay-ambiguous states are neither removed nor relabelled. Accuracy is replay-averaged task accuracy of the executed action. Random = mean of 2000 uniform selections of the same size (expected value = linear interpolation).

## TEXTVQA  n=250  (HELP 35 / HARM 9 / NEUTRAL 189 / AMBIGUOUS 17 = 6.8%)

always-STOP 65.0% @ 0 tok · always-HIGHRES 77.9% @ 307 tok/state

| LOOK budget | PPU v2.2 acc | confidence acc | random acc (±sd) | tok/state (PPU) | AMBIG share of PPU picks |
|---|--:|--:|--:|--:|--:|
| 5% | **66.5%** | 67.0% | 65.6% (±0.6) | 15 | 8% |
| 10% | **67.3%** | 69.9% | 66.3% (±0.8) | 31 | 8% |
| 15% | **67.6%** | 72.1% | 67.0% (±0.9) | 47 | 13% |
| 20% | **69.6%** | 73.3% | 67.6% (±1.0) | 62 | 14% |

## VQAV2  n=250  (HELP 44 / HARM 2 / NEUTRAL 156 / AMBIGUOUS 48 = 19.2%)

always-STOP 42.9% @ 0 tok · always-HIGHRES 65.3% @ 127 tok/state

| LOOK budget | PPU v2.2 acc | confidence acc | random acc (±sd) | tok/state (PPU) | AMBIG share of PPU picks |
|---|--:|--:|--:|--:|--:|
| 5% | **45.7%** | 42.8% | 44.0% (±0.6) | 6 | 33% |
| 10% | **47.3%** | 44.0% | 45.1% (±0.8) | 13 | 32% |
| 15% | **49.5%** | 45.0% | 46.3% (±1.0) | 19 | 26% |
| 20% | **50.7%** | 46.1% | 47.3% (±1.0) | 25 | 28% |
