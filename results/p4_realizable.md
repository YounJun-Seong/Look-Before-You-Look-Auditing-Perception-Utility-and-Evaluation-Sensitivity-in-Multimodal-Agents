# Phi-4 realizable policy — Random vs confidence selector vs oracle

Confidence = Phi-4's own STOP mean token logprob (verified side
collection); selector looks (HIGHRES) below a budget-tuned threshold.
Oracle = budget-constrained cross-replay oracle (all arms). Strict.

## mme  (units=392, STOP 0.247, HIGHRES 0.360 @ 994 tok)

| budget | random | confidence | oracle | conf−random [CI] |
|---|--:|--:|--:|:--|
| 5% | 0.253 | 0.250 | 0.355 | -0.003 [-0.007, +0.003] |
| 10% | 0.259 | 0.258 | 0.355 | -0.001 [-0.014, +0.012] |
| 20% | 0.270 | 0.283 | 0.355 | +0.013 [-0.006, +0.033] |
| 40% | 0.292 | 0.296 | 0.355 | +0.004 [-0.017, +0.025] |
| 100% | 0.360 | 0.360 | 0.355 | +0.000 [+0.000, +0.000] |

## textvqa  (units=400, STOP 0.243, HIGHRES 0.323 @ 308 tok)

| budget | random | confidence | oracle | conf−random [CI] |
|---|--:|--:|--:|:--|
| 5% | 0.247 | 0.244 | 0.340 | -0.003 [-0.005, -0.001] |
| 10% | 0.251 | 0.254 | 0.396 | +0.003 [-0.006, +0.013] |
| 20% | 0.259 | 0.259 | 0.399 | -0.000 [-0.012, +0.013] |
| 40% | 0.275 | 0.283 | 0.399 | +0.008 [-0.008, +0.025] |
| 100% | 0.323 | 0.326 | 0.399 | +0.003 [+0.000, +0.007] |
