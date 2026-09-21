# Does a question-conditioned crop proposal beat a random one?

Cardinality-matched, cross-fitted (choose on R1, score on R2). Where m of k candidates are needed, the value is averaged over every C(k, m) subset. Paired bootstrap 95% CI over samples.

## MME-RealWorld-Lite (n = 6 usable; VLM-proposal counts {0: 78, 1: 14, 3: 6})

| comparison | arm A | arm B | A − B | 95% CI |
|---|--:|--:|--:|---|
| VLM-3 vs Random-3 | 16.7 | 48.5 | **-31.8pp** | [-64.6, +0.0] |
| Grid-5 vs Random-5 | 33.3 | 50.0 | **-16.7pp** | [-50.0, +0.0] |
| VLM-3 vs Grid-3 | 16.7 | 33.3 | **-16.7pp** | [-50.0, +0.0] |
| Structured-8 vs Random-8 | 33.3 | 50.0 | **-16.7pp** | [-50.0, +0.0] |

reference only (naive, chosen and scored on R1): VLM-3 − Random-3 = +0.0pp [+0.0, +0.0]

## TextVQA (n = 4 usable; VLM-proposal counts {0: 65, 1: 30, 2: 1, 3: 4})

| comparison | arm A | arm B | A − B | 95% CI |
|---|--:|--:|--:|---|
| VLM-3 vs Random-3 | 25.0 | 60.3 | **-35.3pp** | [-70.5, +0.0] |
| Grid-5 vs Random-5 | 75.0 | 66.1 | **+8.9pp** | [+0.0, +20.1] |
| VLM-3 vs Grid-3 | 25.0 | 62.5 | **-37.5pp** | [-75.0, +0.0] |
| Structured-8 vs Random-8 | 75.0 | 75.0 | **+0.0pp** | [+0.0, +0.0] |

reference only (naive, chosen and scored on R1): VLM-3 − Random-3 = +1.8pp [+0.0, +5.4]
