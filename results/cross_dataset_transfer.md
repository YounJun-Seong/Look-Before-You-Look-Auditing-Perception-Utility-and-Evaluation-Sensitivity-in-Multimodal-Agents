# (vi) Cross-dataset frozen-selector transfer — TextVQA-frozen selector v2 on VQAv2 held-out (zero inference, disclosed post-hoc)

n = 250; no adaptation of heads, scaler, conf-ECDF or lambda (TextVQA values used verbatim). Mixtures per §8 machinery;
paired 10k bootstrap.

## strict

| budget | selector acc (vtok) | mix-CF acc | sel − mix-CF [95% CI] | sel − mix-insample [95% CI] | oracle (Lagr.) |
|---|---|--:|:--|:--|--:|
| 0.25× | 0.695 (83) | 0.704 | -0.0085 [-0.0507, +0.0331] | -0.0277 [-0.0713, +0.0154] | 0.706 |
| 0.5× | 0.697 (86) | 0.724 | -0.0272 [-0.0710, +0.0150] | -0.0504 [-0.0931, -0.0077] | 0.709 |
| 1× | 0.709 (90) | 0.727 | -0.0187 [-0.0650, +0.0268] | -0.0481 [-0.0958, -0.0009] | 0.709 |

selector action mix: {"0.25": {"CROP": 0.44, "HIGHRES": 0.392, "OCR": 0.132, "STOP": 0.036}, "0.5": {"CROP": 0.444, "HIGHRES": 0.416, "OCR": 0.116, "STOP": 0.024}, "1.0": {"CROP": 0.436, "HIGHRES": 0.448, "OCR": 0.096, "STOP": 0.02}}

## lenient

| budget | selector acc (vtok) | mix-CF acc | sel − mix-CF [95% CI] | sel − mix-insample [95% CI] | oracle (Lagr.) |
|---|---|--:|:--|:--|--:|
| 0.25× | 0.802 (83) | 0.774 | +0.0276 [-0.0050, +0.0621] | +0.0106 [-0.0207, +0.0426] | 0.861 |
| 0.5× | 0.798 (86) | 0.779 | +0.0196 [-0.0153, +0.0559] | +0.0001 [-0.0312, +0.0333] | 0.861 |
| 1× | 0.799 (90) | 0.784 | +0.0148 [-0.0277, +0.0577] | -0.0092 [-0.0511, +0.0333] | 0.861 |

selector action mix: {"0.25": {"CROP": 0.44, "HIGHRES": 0.392, "OCR": 0.132, "STOP": 0.036}, "0.5": {"CROP": 0.444, "HIGHRES": 0.416, "OCR": 0.116, "STOP": 0.024}, "1.0": {"CROP": 0.436, "HIGHRES": 0.448, "OCR": 0.096, "STOP": 0.02}}
