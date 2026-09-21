# Stronger-controller prospective evaluation (strong; n=400)

always-HIGHRES 308 tok. Pool-trained frozen ridge/xgb/mlp (sha 5c60ccab3f7d08b8), frozen v2, frozen mixture; paired 10k bootstrap (seed 20260920).

## strict

always-STOP 0.639 · always-CROP 0.710 · always-OCR 0.625 · always-HIGHRES 0.727

| budget | ridge (vtok) | xgb (vtok) | mlp (vtok) | v2 (vtok) | mixture (vtok) | xgb − mix [CI] | xgb − ridge [CI] | mlp − ridge [CI] | ridge − mix [CI] | random | best-fixed | oracle |
|---|---|---|---|---|---|:--|:--|:--|:--|--:|--:|--:|
| 0.25× | 0.691 (61) | 0.701 (54) | 0.691 (78) | 0.675 (51) | 0.708 (76) | -0.0064 [-0.0359, +0.0231] | +0.0102 [-0.0118, +0.0328] | +0.0000 [-0.0230, +0.0223] | -0.0166 [-0.0445, +0.0118] | 0.662 | 0.639 | 0.812 |
| 0.5× | 0.728 (141) | 0.726 (148) | 0.716 (160) | 0.699 (127) | 0.713 (153) | +0.0129 [-0.0138, +0.0406] | -0.0026 [-0.0203, +0.0157] | -0.0129 [-0.0322, +0.0058] | +0.0155 [-0.0122, +0.0440] | 0.684 | 0.639 | 0.812 |
| 1.0× | 0.729 (210) | 0.731 (216) | 0.727 (225) | 0.704 (216) | 0.704 (241) | +0.0267 [-0.0031, +0.0580] | +0.0020 [-0.0144, +0.0184] | -0.0021 [-0.0204, +0.0161] | +0.0247 [-0.0066, +0.0567] | 0.687 | 0.727 | 0.838 |

Action mixes:

- 0.25×: ridge {'CROP': 0.142, 'HIGHRES': 0.115, 'OCR': 0.375, 'STOP': 0.367}; xgb {'CROP': 0.113, 'HIGHRES': 0.11, 'OCR': 0.3, 'STOP': 0.477}; mlp {'CROP': 0.15, 'HIGHRES': 0.165, 'OCR': 0.268, 'STOP': 0.417}; v2 {'CROP': 0.087, 'HIGHRES': 0.115, 'OCR': 0.525, 'STOP': 0.273}
- 0.5×: ridge {'CROP': 0.198, 'HIGHRES': 0.343, 'OCR': 0.16, 'STOP': 0.3}; xgb {'CROP': 0.253, 'HIGHRES': 0.33, 'OCR': 0.15, 'STOP': 0.268}; mlp {'CROP': 0.26, 'HIGHRES': 0.367, 'OCR': 0.135, 'STOP': 0.237}; v2 {'CROP': 0.155, 'HIGHRES': 0.323, 'OCR': 0.323, 'STOP': 0.2}
- 1.0×: ridge {'CROP': 0.233, 'HIGHRES': 0.545, 'OCR': 0.035, 'STOP': 0.188}; xgb {'CROP': 0.315, 'HIGHRES': 0.515, 'OCR': 0.077, 'STOP': 0.092}; mlp {'CROP': 0.275, 'HIGHRES': 0.568, 'OCR': 0.055, 'STOP': 0.102}; v2 {'CROP': 0.175, 'HIGHRES': 0.598, 'OCR': 0.18, 'STOP': 0.048}

## lenient

always-STOP 0.775 · always-CROP 0.782 · always-OCR 0.758 · always-HIGHRES 0.841

| budget | ridge (vtok) | xgb (vtok) | mlp (vtok) | v2 (vtok) | mixture (vtok) | xgb − mix [CI] | xgb − ridge [CI] | mlp − ridge [CI] | ridge − mix [CI] | random | best-fixed | oracle |
|---|---|---|---|---|---|:--|:--|:--|:--|--:|--:|--:|
| 0.25× | 0.786 (61) | 0.802 (54) | 0.790 (78) | 0.779 (51) | 0.784 (76) | +0.0180 [-0.0106, +0.0476] | +0.0162 [-0.0053, +0.0375] | +0.0037 [-0.0179, +0.0251] | +0.0018 [-0.0258, +0.0293] | 0.784 | 0.775 | 0.903 |
| 0.5× | 0.817 (141) | 0.818 (148) | 0.811 (160) | 0.803 (127) | 0.785 (153) | +0.0332 [+0.0066, +0.0608] | +0.0014 [-0.0158, +0.0188] | -0.0064 [-0.0232, +0.0096] | +0.0318 [+0.0047, +0.0589] | 0.793 | 0.775 | 0.903 |
| 1.0× | 0.829 (210) | 0.825 (216) | 0.824 (225) | 0.816 (216) | 0.781 (241) | +0.0441 [+0.0136, +0.0752] | -0.0034 [-0.0173, +0.0103] | -0.0047 [-0.0182, +0.0083] | +0.0476 [+0.0180, +0.0781] | 0.794 | 0.841 | 0.909 |

Action mixes:

- 0.25×: ridge {'CROP': 0.142, 'HIGHRES': 0.115, 'OCR': 0.375, 'STOP': 0.367}; xgb {'CROP': 0.113, 'HIGHRES': 0.11, 'OCR': 0.3, 'STOP': 0.477}; mlp {'CROP': 0.15, 'HIGHRES': 0.165, 'OCR': 0.268, 'STOP': 0.417}; v2 {'CROP': 0.087, 'HIGHRES': 0.115, 'OCR': 0.525, 'STOP': 0.273}
- 0.5×: ridge {'CROP': 0.198, 'HIGHRES': 0.343, 'OCR': 0.16, 'STOP': 0.3}; xgb {'CROP': 0.253, 'HIGHRES': 0.33, 'OCR': 0.15, 'STOP': 0.268}; mlp {'CROP': 0.26, 'HIGHRES': 0.367, 'OCR': 0.135, 'STOP': 0.237}; v2 {'CROP': 0.155, 'HIGHRES': 0.323, 'OCR': 0.323, 'STOP': 0.2}
- 1.0×: ridge {'CROP': 0.233, 'HIGHRES': 0.545, 'OCR': 0.035, 'STOP': 0.188}; xgb {'CROP': 0.315, 'HIGHRES': 0.515, 'OCR': 0.077, 'STOP': 0.092}; mlp {'CROP': 0.275, 'HIGHRES': 0.568, 'OCR': 0.055, 'STOP': 0.102}; v2 {'CROP': 0.175, 'HIGHRES': 0.598, 'OCR': 0.18, 'STOP': 0.048}

**Primary endpoint (xgb − mixture, ½× strict): +0.0129 [-0.0138, +0.0406] → FAIL (CI includes 0)**
