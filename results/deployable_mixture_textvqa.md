# Deployable mixture (dev-fitted) vs frozen selector — TextVQA strict

Weights fitted on Gate 0 dev arm means (n=200 states),
applied unchanged to the 250 held-out states. Arms matched by fixed
crop box; dev-only arms dropped: none.
Dev always-HIGHRES 308 tok; held-out 307 tok.

| budget | dev-fitted mixture | dev acc (fit) | held-out acc | held-out vtok | multi acc | multi − mixture [95% CI] |
|---|---|--:|--:|--:|--:|:--|
| 0.25× | 0.88·CROP@0.3500,0.3500,0.6500,0.6500 + 0.12·OCR | 0.702 | 0.734 | 75 | 0.745 | +0.0102 [-0.0248, +0.0457] |
| 0.5× | 0.57·CROP@0.3500,0.3500,0.6500,0.6500 + 0.43·CROP@0.4500,0.0000,1.0000,0.5500 | 0.717 | 0.744 | 152 | 0.774 | +0.0296 [-0.0032, +0.0634] |
| 1× | 1.00·CROP@0.4500,0.0000,1.0000,0.5500 | 0.727 | 0.746 | 240 | 0.774 | +0.0279 [-0.0087, +0.0645] |