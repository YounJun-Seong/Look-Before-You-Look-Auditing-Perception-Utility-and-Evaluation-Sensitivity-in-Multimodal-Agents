# Prospective selector-vs-mixture evaluation (preregistered, 1 pass)

n = 600 fresh TextVQA states; always-HIGHRES 307 tok. Frozen
selector v2 + frozen dev-fitted mixture; paired 10k bootstrap.

## strict

always-STOP 0.631 · always-HIGHRES 0.712 · always-CROP 0.711 · always-OCR 0.640

| budget | selector acc (vtok) | mixture acc (vtok) | sel − mix [95% CI] | random | best-fixed-at-B | oracle (Lagr.) |
|---|---|---|:--|--:|--:|--:|
| 0.25× | 0.696 (53) | 0.695 (77) | +0.0018 [-0.0232, +0.0276] | 0.651 | 0.631 | 0.818 |
| 0.5× | 0.720 (129) | 0.711 (154) | +0.0090 [-0.0159, +0.0339] | 0.672 | 0.631 | 0.818 |
| 1× | 0.727 (212) | 0.722 (241) | +0.0051 [-0.0219, +0.0318] | 0.712 | 0.712 | 0.844 |

selector action mix: {"0.25": {"CROP": 0.128, "HIGHRES": 0.095, "OCR": 0.543, "STOP": 0.233}, "0.5": {"CROP": 0.203, "HIGHRES": 0.3, "OCR": 0.352, "STOP": 0.145}, "1.0": {"CROP": 0.225, "HIGHRES": 0.558, "OCR": 0.185, "STOP": 0.032}}

## lenient

always-STOP 0.751 · always-HIGHRES 0.830 · always-CROP 0.771 · always-OCR 0.770

| budget | selector acc (vtok) | mixture acc (vtok) | sel − mix [95% CI] | random | best-fixed-at-B | oracle (Lagr.) |
|---|---|---|:--|--:|--:|--:|
| 0.25× | 0.790 (53) | 0.760 (77) | +0.0298 [+0.0076, +0.0523] | 0.770 | 0.751 | 0.897 |
| 0.5× | 0.817 (129) | 0.771 (154) | +0.0458 [+0.0242, +0.0678] | 0.790 | 0.751 | 0.897 |
| 1× | 0.819 (212) | 0.789 (241) | +0.0303 [+0.0074, +0.0545] | 0.830 | 0.830 | 0.901 |

selector action mix: {"0.25": {"CROP": 0.128, "HIGHRES": 0.095, "OCR": 0.543, "STOP": 0.233}, "0.5": {"CROP": 0.203, "HIGHRES": 0.3, "OCR": 0.352, "STOP": 0.145}, "1.0": {"CROP": 0.225, "HIGHRES": 0.558, "OCR": 0.185, "STOP": 0.032}}
