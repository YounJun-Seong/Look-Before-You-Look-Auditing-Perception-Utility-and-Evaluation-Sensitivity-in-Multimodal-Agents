# Cost-aware sequential controller v2 on the 300-state set (disclosed secondary; n=300)

Frozen on dev-100 (sha in models/seq_controller_v2_frozen.json;
a1-confidence regeneration byte-match 379/400, disclosed). Paired
10k bootstrap vs the frozen mixture (= always→TR).

## strict

| a1 | policy | acc | Δvtok | look | vs always→TR [95% CI] |
|---|---|--:|--:|--:|:--|
| center | stop_after_a1 | 0.741 | 0 | 0.00 | -0.0156 [-0.0361, +0.0050] |
| center | always_tr(mixture) | 0.756 | 241 | 1.00 | +0.0000 [+0.0000, +0.0000] |
| center | v1_controller | 0.739 | 284 | 1.00 | -0.0167 [-0.0356, +0.0017] |
| center | v2_lam0 | 0.750 | 215 | 0.83 | -0.0061 [-0.0178, +0.0056] |
| center | v2_lamB | 0.750 | 215 | 0.83 | -0.0061 [-0.0178, +0.0056] |
| ocr | stop_after_a1 | 0.614 | 0 | 0.00 | -0.0822 [-0.1106, -0.0539] |
| ocr | always_tr(mixture) | 0.696 | 241 | 1.00 | +0.0000 [+0.0000, +0.0000] |
| ocr | v1_controller | 0.697 | 263 | 1.00 | +0.0006 [-0.0139, +0.0150] |
| ocr | v2_lam0 | 0.701 | 230 | 0.85 | +0.0044 [-0.0183, +0.0272] |
| ocr | v2_lamB | 0.680 | 80 | 0.30 | -0.0161 [-0.0422, +0.0100] |

## lenient

| a1 | policy | acc | Δvtok | look | vs always→TR [95% CI] |
|---|---|--:|--:|--:|:--|
| center | stop_after_a1 | 0.808 | 0 | 0.00 | -0.0072 [-0.0244, +0.0106] |
| center | always_tr(mixture) | 0.815 | 241 | 1.00 | +0.0000 [+0.0000, +0.0000] |
| center | v1_controller | 0.803 | 284 | 1.00 | -0.0122 [-0.0289, +0.0044] |
| center | v2_lam0 | 0.813 | 215 | 0.83 | -0.0017 [-0.0122, +0.0089] |
| center | v2_lamB | 0.813 | 215 | 0.83 | -0.0017 [-0.0122, +0.0089] |
| ocr | stop_after_a1 | 0.741 | 0 | 0.00 | -0.0317 [-0.0550, -0.0083] |
| ocr | always_tr(mixture) | 0.773 | 241 | 1.00 | +0.0000 [+0.0000, +0.0000] |
| ocr | v1_controller | 0.781 | 263 | 1.00 | +0.0083 [-0.0050, +0.0233] |
| ocr | v2_lam0 | 0.799 | 230 | 0.85 | +0.0267 [+0.0050, +0.0483] |
| ocr | v2_lamB | 0.789 | 80 | 0.30 | +0.0161 [-0.0061, +0.0389] |
