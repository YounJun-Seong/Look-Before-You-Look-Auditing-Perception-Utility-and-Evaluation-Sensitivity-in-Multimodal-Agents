# Optimal state-independent randomized fixed mixture vs frozen selector

max_p sum p(a)ybar(a) s.t. sum p(a)cbar(a) <= B over fixed arms
{STOP, grid CROP slots, uniform CROP, OCR, HIGHRES}; weights fitted
on the evaluation sample itself (favors the baseline). Paired
bootstrap of (frozen multi-action - mixture), 10k resamples.

## vqav2 / strict  (n=250, always-HIGHRES 126 tok)

| budget | mixture (weights, in-sample) | mix acc | mix acc (CF) | multi acc | multi − mix [CI] | multi − mix-CF [CI] |
|---|---|--:|--:|--:|:--|:--|
| 0.25× | 0.94·CROP_slot5 + 0.06·STOP | 0.723 | 0.704 | 0.551 | -0.1718 [-0.2250, -0.1206] | -0.1526 [-0.2037, -0.1023] |
| 0.5× | 0.60·CROP_slot5 + 0.40·CROP_slot3 | 0.747 | 0.724 | 0.652 | -0.0948 [-0.1395, -0.0497] | -0.0715 [-0.1171, -0.0272] |
| 1× | 1.00·CROP_slot3 | 0.757 | 0.727 | 0.715 | -0.0416 [-0.0886, +0.0056] | -0.0123 [-0.0568, +0.0322] |

## vqav2 / lenient  (n=250, always-HIGHRES 126 tok)

| budget | mixture (weights, in-sample) | mix acc | mix acc (CF) | multi acc | multi − mix [CI] | multi − mix-CF [CI] |
|---|---|--:|--:|--:|:--|:--|
| 0.25× | 0.71·OCR + 0.29·CROP_slot3 | 0.791 | 0.774 | 0.784 | -0.0070 [-0.0241, +0.0103] | +0.0100 [-0.0170, +0.0377] |
| 0.5× | 0.58·CROP_slot3 + 0.42·OCR | 0.798 | 0.779 | 0.792 | -0.0057 [-0.0334, +0.0219] | +0.0138 [-0.0172, +0.0452] |
| 1× | 1.00·CROP_slot3 | 0.808 | 0.784 | 0.808 | +0.0002 [-0.0407, +0.0406] | +0.0242 [-0.0167, +0.0645] |

## textvqa / strict  (n=250, always-HIGHRES 307 tok)

| budget | mixture (weights, in-sample) | mix acc | mix acc (CF) | multi acc | multi − mix [CI] | multi − mix-CF [CI] |
|---|---|--:|--:|--:|:--|:--|
| 0.25× | 0.90·CROP_slot5 + 0.10·OCR | 0.736 | 0.725 | 0.745 | +0.0088 [-0.0268, +0.0450] | +0.0199 [-0.0167, +0.0564] |
| 0.5× | 0.54·CROP_slot5 + 0.46·CROP_slot0 | 0.762 | 0.756 | 0.774 | +0.0117 [-0.0218, +0.0460] | +0.0179 [-0.0123, +0.0491] |
| 1× | 1.00·CROP_slot0 | 0.785 | 0.785 | 0.774 | -0.0107 [-0.0452, +0.0236] | -0.0107 [-0.0452, +0.0236] |
