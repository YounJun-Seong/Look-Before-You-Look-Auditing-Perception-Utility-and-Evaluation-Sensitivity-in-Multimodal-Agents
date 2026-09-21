# (v) Cross-reader selector transfer — Qwen-trained selector v2 on Phi-4 states (zero inference, disclosed post-hoc)

Same states/views, Phi-4 outcomes + Phi-4 confidence (verified side
collection). verbatim = Qwen ECDF/λ unchanged; recalibrated = conf
rank within Phi-4's own distribution. Strict; paired 10k bootstrap.

## mme  (n=196, always-STOP 0.247, always-HIGHRES vtok 994)

| budget | variant | selector acc (vtok) | mix-CF | sel − mix-CF [95% CI] | sel − STOP [95% CI] |
|---|---|---|--:|:--|:--|
| 0.25× | verbatim | 0.252 (1) | 0.332 | -0.0798 [-0.1284, -0.0311] | +0.0047 [-0.0386, +0.0507] |
| 0.25× | recalibrated | 0.250 (0) | 0.332 | -0.0820 [-0.1326, -0.0314] | +0.0026 [-0.0408, +0.0485] |
| 0.5× | verbatim | 0.253 (12) | 0.341 | -0.0880 [-0.1337, -0.0413] | +0.0058 [-0.0376, +0.0513] |
| 0.5× | recalibrated | 0.253 (6) | 0.341 | -0.0880 [-0.1337, -0.0413] | +0.0058 [-0.0376, +0.0513] |
| 1× | verbatim | 0.256 (73) | 0.360 | -0.1039 [-0.1544, -0.0525] | +0.0084 [-0.0359, +0.0534] |
| 1× | recalibrated | 0.258 (60) | 0.360 | -0.1015 [-0.1530, -0.0496] | +0.0107 [-0.0327, +0.0561] |

## textvqa  (n=200, always-STOP 0.243, always-HIGHRES vtok 308)

| budget | variant | selector acc (vtok) | mix-CF | sel − mix-CF [95% CI] | sel − STOP [95% CI] |
|---|---|---|--:|:--|:--|
| 0.25× | verbatim | 0.345 (38) | 0.331 | +0.0144 [-0.0018, +0.0297] | +0.1021 [+0.0614, +0.1457] |
| 0.25× | recalibrated | 0.344 (41) | 0.331 | +0.0128 [-0.0029, +0.0276] | +0.1005 [+0.0594, +0.1443] |
| 0.5× | verbatim | 0.340 (81) | 0.320 | +0.0198 [-0.0085, +0.0479] | +0.0968 [+0.0589, +0.1366] |
| 0.5× | recalibrated | 0.347 (94) | 0.320 | +0.0268 [-0.0032, +0.0566] | +0.1038 [+0.0654, +0.1448] |
| 1× | verbatim | 0.328 (174) | 0.308 | +0.0196 [-0.0195, +0.0595] | +0.0846 [+0.0477, +0.1243] |
| 1× | recalibrated | 0.323 (182) | 0.308 | +0.0143 [-0.0253, +0.0556] | +0.0793 [+0.0435, +0.1181] |
