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
| CROP_grid | 392 | 0.158 | 0.449 / 0.169 | 0.362 / 0.126 | 0.022 | 234 |
| CROP_question | 118 | 0.028 | 0.200 / 0.164 | 0.175 / 0.159 | 0.008 | 144 |
| CROP_random | 196 | 0.046 | 0.295 / 0.129 | 0.215 / 0.100 | 0.029 | 225 |
| OCR | 392 | 0.028 | 0.080 / 0.080 | 0.091 / 0.091 | 0.031 | 0 |
| HIGHRES | 392 | 0.051 | 0.145 / 0.145 | 0.185 / 0.185 | 0.069 | 994 |
| any_additional | 392 | 0.163 | 0.464 / 0.157 | 0.429 / 0.127 | 0.027 | 281 |

## textvqa

| family | n(units) | DMG (uncond.) | P(harm\|STOP✓) any / single | P(help\|STOP✗) any / single | net Δscore | vtok |
|---|--:|--:|:--|:--|--:|--:|
| CROP_grid | 400 | 0.090 | 0.154 / 0.049 | 0.729 / 0.313 | 0.075 | 189 |
| CROP_question | 132 | 0.003 | 0.015 / 0.015 | 0.340 / 0.308 | 0.109 | 81 |
| CROP_random | 200 | 0.035 | 0.123 / 0.042 | 0.716 / 0.313 | 0.082 | 182 |
| OCR | 400 | 0.050 | 0.085 / 0.085 | 0.241 / 0.241 | 0.028 | 0 |
| HIGHRES | 400 | 0.025 | 0.043 / 0.043 | 0.421 / 0.421 | 0.094 | 308 |
| any_additional | 400 | 0.130 | 0.222 / 0.051 | 0.782 / 0.314 | 0.073 | 177 |
