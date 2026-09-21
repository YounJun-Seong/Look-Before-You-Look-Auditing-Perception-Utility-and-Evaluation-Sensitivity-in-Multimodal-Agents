# Latency at the token-matched operating points (held-out, strict)

Mean added latency per state (generation + OCR tool time, seconds) of
the operating points reported in the token-budget tables. Same frozen
numbers, different cost axis; the two axes are never summed.

## vqav2

| policy @ budget | acc | vtok | latency (s) |
|---|--:|--:|--:|
| always STOP | 0.448 | 0 | 0.00 |
| always HIGHRES | 0.607 | 126 | 0.60 |
| always OCR | 0.439 | 0 | 25.50 |
| always CROP | 0.741 | 77 | 0.29 |
| confidence @ 0.25× | 0.477 | 29 | 0.14 |
| v22_gate @ 0.25× | 0.509 | 28 | 0.14 |
| multiaction @ 0.25× | 0.551 | 22 | 11.06 |
| confidence @ 0.5× | 0.504 | 60 | 0.28 |
| v22_gate @ 0.5× | 0.595 | 62 | 0.31 |
| multiaction @ 0.5× | 0.652 | 58 | 8.76 |
| confidence @ 1× | 0.607 | 126 | 0.60 |
| v22_gate @ 1× | 0.610 | 113 | 0.54 |
| multiaction @ 1× | 0.715 | 90 | 3.09 |
| cross-replay oracle | 0.731 | 39 | 2.82 |

## textvqa

| policy @ budget | acc | vtok | latency (s) |
|---|--:|--:|--:|
| always STOP | 0.675 | 0 | 0.00 |
| always HIGHRES | 0.777 | 307 | 0.67 |
| always OCR | 0.676 | 0 | 61.19 |
| always CROP | 0.745 | 181 | 0.10 |
| confidence @ 0.25× | 0.736 | 70 | 0.15 |
| v22_gate @ 0.25× | 0.729 | 70 | 0.16 |
| multiaction @ 0.25× | 0.745 | 67 | 29.01 |
| confidence @ 0.5× | 0.765 | 153 | 0.33 |
| v22_gate @ 0.5× | 0.764 | 153 | 0.34 |
| multiaction @ 0.5× | 0.774 | 152 | 18.53 |
| confidence @ 1× | 0.782 | 245 | 0.54 |
| v22_gate @ 1× | 0.779 | 298 | 0.65 |
| multiaction @ 1× | 0.774 | 207 | 14.27 |
| cross-replay oracle | 0.842 | 46 | 3.57 |
