# Fresh prospective lockbox — pre-registered verdict

| | prediction | result | evidence |
|---|---|:--:|---|
| P1 | F1 strict: ROC<0.55, verbose>25%, format-flip HELP>=50% | **PASS** | ROC 0.428, verbose 40%, flips 77% |
| P2 | F1 lenient ROC >= F1 strict ROC + 0.15 | **FAIL** | 0.428 -> 0.526 (Δ +0.098) |
| P3 | F2 strict: verbose<10%, STOP >= F1 STOP+20pp, ROC>0.60, CI lo>0.5 | **PASS** | verbose 3%, STOP 44.8->78.0, ROC 0.897 [0.84, 0.95] |
| P4 | F3, F4 strict ROC > 0.70 and |F3-F4| < 0.10 | **PASS** | F3 0.786, F4 0.815 |
| P5 | lenient: |F1 - F2| < 0.15 | **FAIL** | F1 0.526, F2 0.906 |

## Full table (strict / lenient)

| arm | scorer | STOP | HIGHRES | verbose | HELP (prev.) | flips | conf ROC [CI] | lift |
|---|---|--:|--:|--:|--:|--:|:--:|--:|
| F1 VQAv2 original | strict | 44.8 | 60.7 | 40% | 30 (16.8%) | 77% | 0.428 [0.32, 0.54] | 0.80× |
| F1 VQAv2 original | lenient | 75.8 | 76.3 | 40% | 19 (8.6%) | 47% | 0.526 [0.38, 0.66] | 1.22× |
| F2 VQAv2 one-word | strict | 78.0 | 80.7 | 3% | 14 (6.1%) | 7% | 0.897 [0.84, 0.95] | 4.65× |
| F2 VQAv2 one-word | lenient | 77.8 | 80.7 | 3% | 15 (6.6%) | 7% | 0.906 [0.85, 0.95] | 5.07× |
| F3 TextVQA original | strict | 67.5 | 77.7 | 43% | 29 (12.9%) | 21% | 0.786 [0.71, 0.86] | 3.21× |
| F3 TextVQA original | lenient | 79.5 | 86.8 | 43% | 26 (11.5%) | 8% | 0.827 [0.75, 0.90] | 4.35× |
| F4 TextVQA one-word | strict | 76.4 | 82.4 | 26% | 18 (8.0%) | 6% | 0.815 [0.72, 0.90] | 2.78× |
| F4 TextVQA one-word | lenient | 78.2 | 83.6 | 26% | 19 (8.3%) | 5% | 0.822 [0.73, 0.90] | 3.17× |