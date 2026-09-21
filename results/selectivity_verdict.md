# Is question-conditioned crop selection worth anything?

Per sample, k = number of valid VLM proposals. Every arm is matched to that k, chosen on R1 and scored on R2; arms with more than k candidates are averaged over all C(n, k) subsets. Paired bootstrap 95% CI.

## MME-RealWorld-Lite

proposal failure rate (k = 0): **34.2%** (67/196); usable samples n = 129, k distribution {1: 99, 2: 4, 3: 26}

| comparison | arm A | arm B | A − B | 95% CI | verdict |
|---|--:|--:|--:|---|---|
| VLM-k vs MatchedRandom-k | 39.5 | 36.4 | **+3.1pp** | [-2.3, +8.5] | n.s. |
| VLM-k vs Grid-k | 39.5 | 41.0 | **-1.5pp** | [-6.9, +4.1] | n.s. |
| VLM-k vs FULL | 39.5 | 41.9 | **-2.3pp** | [-8.5, +3.9] | n.s. |
| VLM-k vs Random-k (unmatched) | 34.9 | 32.5 | **+2.4pp** | [-6.5, +11.9] | n.s. |

reference: STOP 39.5, FULL 41.9

| subset | n | VLM − MatchedRandom |
|---|--:|--:|
| coarse wrong | 78 | +5.1pp |
| coarse right | 51 | +0.0pp |

## TextVQA

proposal failure rate (k = 0): **20.5%** (41/200); usable samples n = 159, k distribution {1: 133, 2: 5, 3: 21}

| comparison | arm A | arm B | A − B | 95% CI | verdict |
|---|--:|--:|--:|---|---|
| VLM-k vs MatchedRandom-k | 72.7 | 67.3 | **+5.5pp** | [+0.6, +10.3] | significant |
| VLM-k vs Grid-k | 72.7 | 70.0 | **+2.7pp** | [-1.4, +6.9] | n.s. |
| VLM-k vs FULL | 72.7 | 74.4 | **-1.7pp** | [-8.2, +4.6] | n.s. |
| VLM-k vs Random-k (unmatched) | 71.4 | 69.4 | **+1.9pp** | [-4.7, +8.5] | n.s. |

reference: STOP 57.4, FULL 74.4

| subset | n | VLM − MatchedRandom |
|---|--:|--:|
| coarse wrong | 68 | +11.8pp |
| coarse right | 91 | +0.7pp |

## Pooled across datasets

| comparison | A − B | 95% CI |
|---|--:|---|
| VLM-k vs MatchedRandom-k | **+4.4pp** | [+0.9, +8.2] |
| VLM-k vs Grid-k | **+0.8pp** | [-2.5, +4.2] |
| VLM-k vs FULL | **-2.0pp** | [-6.4, +2.5] |
| VLM-k vs Random-k (unmatched) | **+2.2pp** | [-3.2, +7.7] |
