# Feature-group baselines on the lockbox (single pass, trained on development)

All trained variants: v2.2 recipe verbatim (same 601 development states, same hyper-parameters, frozen development confidence CDF); only the feature set changes. v2.2 itself is untouched. Budget accuracies are replay-averaged task accuracy when the top-b fraction of states by score is sent to HIGHRES.

## TEXTVQA  (n=233, prevalence 15.0%, always-STOP 66.9% / always-HIGHRES 78.0%)

| variant | ROC | PR/prev | lift@1× | 95% CI | acc@10% | acc@20% | HELP recall@20% | dev OOF ROC (tv / vqa) |
|---|--:|--:|--:|:--:|--:|--:|--:|:--:|
| confidence only (no training) | 0.796 | 2.76× | 3.23× | [2.16, 4.08] | 71.5% | 74.9% | 54.3% | — |
| question surface only | 0.458 | 0.90× | 0.76× | [0.17, 1.46] | 67.6% | 69.0% | 14.3% | 0.398 / 0.528 |
| answer surface only | 0.549 | 1.15× | 1.14× | [0.27, 2.05] | 67.7% | 69.7% | 25.7% | 0.537 / 0.692 |
| question + answer, no confidence | 0.548 | 1.19× | 1.14× | [0.51, 2.07] | 68.4% | 69.9% | 25.7% | 0.530 / 0.698 |
| full recipe, retrained (sanity) | 0.669 | 1.59× | 1.52× | [0.62, 2.36] | 69.0% | 70.7% | 31.4% | 0.603 / 0.688 |
| full v2.2 (frozen) | 0.669 | 1.59× | 1.52× | [0.62, 2.36] | 69.0% | 70.7% | 31.4% | — |

## VQAV2  (n=202, prevalence 21.8%, always-STOP 44.2% / always-HIGHRES 65.2%)

| variant | ROC | PR/prev | lift@1× | 95% CI | acc@10% | acc@20% | HELP recall@20% | dev OOF ROC (tv / vqa) |
|---|--:|--:|--:|:--:|--:|--:|--:|:--:|
| confidence only (no training) | 0.423 | 0.86× | 0.73× | [0.26, 1.17] | 44.5% | 47.2% | 13.6% | — |
| question surface only | 0.526 | 1.19× | 1.15× | [0.59, 1.63] | 46.8% | 48.9% | 20.5% | 0.398 / 0.528 |
| answer surface only | 0.733 | 2.14× | 2.09× | [1.49, 2.62] | 49.2% | 53.1% | 40.9% | 0.537 / 0.692 |
| question + answer, no confidence | 0.723 | 2.13× | 1.98× | [1.43, 2.68] | 49.2% | 52.5% | 38.6% | 0.530 / 0.698 |
| full recipe, retrained (sanity) | 0.703 | 1.96× | 1.77× | [1.20, 2.35] | 48.7% | 51.7% | 34.1% | 0.603 / 0.688 |
| full v2.2 (frozen) | 0.703 | 1.96× | 1.77× | [1.20, 2.39] | 48.7% | 51.7% | 34.1% | — |
