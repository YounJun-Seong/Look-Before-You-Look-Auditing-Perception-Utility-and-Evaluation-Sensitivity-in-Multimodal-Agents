# Confidence → perceptual-need relationship across backbones

Confidence-based uncertainty (−mean token log-prob of the coarse STOP answer) ranking replicate-consistent HELP states. Labels are regenerated per backbone from its own STOP/HIGHRES rollouts on the same lockbox state ids and pixel views. No training.

| backbone | regime | n (eval) | prev. | STOP acc | HIGHRES acc | ROC [95% CI] | PR/prev | lift@1× [95% CI] |
|---|---|--:|--:|--:|--:|:--:|--:|:--:|
| Qwen2.5-VL-7B | textvqa | 250 (233) | 15.0% | 65.0% | 77.9% | **0.796** [0.71, 0.87] | 2.76× | **3.23×** [2.15, 4.08] |
| Qwen2.5-VL-7B | vqav2 | 250 (202) | 21.8% | 42.9% | 65.3% | **0.423** [0.33, 0.52] | 0.86× | **0.73×** [0.28, 1.17] |
| Phi-4-MM-5.6B | textvqa | 250 (223) | 18.8% | 66.4% | 84.7% | **0.839** [0.78, 0.90] | 2.58× | **2.78×** [2.15, 3.54] |
| Phi-4-MM-5.6B | vqav2 | 250 (234) | 4.3% | 76.1% | 77.3% | **0.818** [0.64, 0.96] | 7.43× | **7.02×** [1.80, 14.62] |
| Phi-4-MM-5.6B (coarse 16) | vqav2 | 250 (232) | 9.1% | 71.8% | 78.1% | **0.814** [0.72, 0.89] | 4.86× | **4.21×** [1.92, 6.63] |
| Qwen2.5-VL-7B (one-word prompt) | vqav2 | 250 (236) | 8.5% | 78.1% | 83.7% | **0.715** [0.60, 0.82] | 2.94× | **2.36×** [0.51, 4.72] |

## Full-population LOOK budgets, confidence selector (AMBIGUOUS included)

| backbone | regime | always-STOP | 5% | 10% | 15% | 20% | always-HIGHRES |
|---|---|--:|--:|--:|--:|--:|--:|
| Qwen2.5-VL-7B | textvqa | 65.0% | 67.0% (rand 65.6) | 69.9% (rand 66.3) | 72.1% (rand 67.0) | 73.3% (rand 67.6) | 77.9% |
| Qwen2.5-VL-7B | vqav2 | 42.9% | 42.8% (rand 43.9) | 44.0% (rand 45.1) | 45.0% (rand 46.3) | 46.1% (rand 47.3) | 65.3% |
| Phi-4-MM-5.6B | textvqa | 66.4% | 68.3% (rand 67.3) | 71.7% (rand 68.2) | 74.3% (rand 69.2) | 74.7% (rand 70.1) | 84.7% |
| Phi-4-MM-5.6B | vqav2 | 76.1% | 77.0% (rand 76.1) | 77.5% (rand 76.2) | 78.1% (rand 76.2) | 77.5% (rand 76.3) | 77.3% |
| Phi-4-MM-5.6B (coarse 16) | vqav2 | 71.8% | 73.9% (rand 72.1) | 74.1% (rand 72.4) | 73.5% (rand 72.8) | 74.4% (rand 73.1) | 78.1% |
| Qwen2.5-VL-7B (one-word prompt) | vqav2 | 78.1% | 79.0% (rand 78.3) | 79.3% (rand 78.6) | 79.3% (rand 78.9) | 79.7% (rand 79.2) | 83.7% |

- Qwen2.5-VL-7B: ROC gap TextVQA − VQAv2 = **+0.373**; VQAv2 confidence below chance, lift < 1×.
- Phi-4-MM-5.6B: ROC gap TextVQA − VQAv2 = **+0.021**; VQAv2 confidence above chance, lift ≥ 1×.