# Per-action damage decomposition (post-hoc, Gate 0 logs)

Unit = state x replay with a logged STOP. `any` = >=1 action in the
family flips (the DMG-style existential); `single` = uniform-random
single action in the family (deployable rate). Scores strict as logged.
Note: unit here is state x replay; the paper's pooled DMG (13.8/13.0)
uses per-direction pooling in ppu/audit.py — textvqa reproduces exactly,
mme differs by 2.5pp under the unit definition (both are reported).

## mme

| family | n(units) | DMG (uncond.) | P(harm\|STOP✓) any / single | P(help\|STOP✗) any / single | net Δscore | vtok |
|---|--:|--:|:--|:--|--:|--:|
| CROP_grid | 392 | 0.092 | 0.371 / 0.185 | 0.275 / 0.128 | 0.051 | 241 |
| CROP_question | 392 | 0.084 | 0.340 / 0.268 | 0.149 / 0.096 | 0.006 | 173 |
| CROP_random | 196 | 0.033 | 0.210 / 0.109 | 0.209 / 0.097 | 0.032 | 234 |
| OCR | 392 | 0.077 | 0.309 / 0.309 | 0.102 / 0.102 | 0.000 | 0 |
| HIGHRES | 392 | 0.028 | 0.113 / 0.113 | 0.186 / 0.186 | 0.112 | 994 |
| any_additional | 392 | 0.115 | 0.464 / 0.207 | 0.336 / 0.124 | 0.042 | 280 |

## textvqa

| family | n(units) | DMG (uncond.) | P(harm\|STOP✓) any / single | P(help\|STOP✗) any / single | net Δscore | vtok |
|---|--:|--:|:--|:--|--:|--:|
| CROP_grid | 400 | 0.065 | 0.306 / 0.110 | 0.265 / 0.111 | 0.049 | 228 |
| CROP_question | 396 | 0.018 | 0.084 / 0.072 | 0.083 / 0.064 | 0.025 | 82 |
| CROP_random | 200 | 0.050 | 0.476 / 0.199 | 0.366 / 0.129 | 0.035 | 196 |
| OCR | 400 | 0.022 | 0.106 / 0.106 | 0.182 / 0.182 | 0.098 | 0 |
| HIGHRES | 400 | 0.013 | 0.059 / 0.059 | 0.141 / 0.141 | 0.080 | 308 |
| any_additional | 400 | 0.077 | 0.365 / 0.096 | 0.378 / 0.109 | 0.051 | 177 |
