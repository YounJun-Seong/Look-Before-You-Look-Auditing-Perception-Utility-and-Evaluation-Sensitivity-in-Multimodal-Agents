# (A) Realized-cost matched mixture vs frozen selector (post-hoc)

Dev-fitted mixture re-solved at the selector's REALIZED mean token
cost per budget (weights from Gate 0 dev arm means only). Paired
10k bootstrap on the 600 prospective states.

## strict

| budget | matched B (tok) | mixture (weights) | mixture acc (vtok) | selector acc (vtok) | sel − mix [95% CI] |
|---|--:|---|---|---|:--|
| 0.25× | 53 | 0.60·C@0.3500,0.3500,0.6500,0.6500 + 0.40·OCR | 0.677 (52) | 0.696 (53) | +0.0190 [-0.0011, +0.0400] |
| 0.5× | 129 | 0.73·C@0.3500,0.3500,0.6500,0.6500 + 0.27·C@0.4500,0.0000,1.0000,0.5500 | 0.708 (129) | 0.720 (129) | +0.0122 [-0.0136, +0.0378] |
| 1× | 212 | 0.81·C@0.4500,0.0000,1.0000,0.5500 + 0.19·C@0.3500,0.3500,0.6500,0.6500 | 0.718 (212) | 0.727 (212) | +0.0088 [-0.0166, +0.0340] |

## lenient

| budget | matched B (tok) | mixture (weights) | mixture acc (vtok) | selector acc (vtok) | sel − mix [95% CI] |
|---|--:|---|---|---|:--|
| 0.25× | 53 | 0.60·C@0.3500,0.3500,0.6500,0.6500 + 0.40·OCR | 0.763 (52) | 0.790 (53) | +0.0267 [+0.0087, +0.0449] |
| 0.5× | 129 | 0.73·C@0.3500,0.3500,0.6500,0.6500 + 0.27·C@0.4500,0.0000,1.0000,0.5500 | 0.766 (129) | 0.817 (129) | +0.0507 [+0.0283, +0.0738] |
| 1× | 212 | 0.81·C@0.4500,0.0000,1.0000,0.5500 + 0.19·C@0.3500,0.3500,0.6500,0.6500 | 0.783 (212) | 0.819 (212) | +0.0361 [+0.0143, +0.0586] |
