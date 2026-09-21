# Second-reader replication — Qwen2.5-VL-7B vs Phi-4-multimodal

Frozen `ppu.audit` on identical states, views and replay procedure
(prereg: paper/PREREG_SECOND_READER.md). All numbers %, strict scorer.

## mme  (n = 196 / 196)

| metric | Qwen | Phi-4 |
|---|--:|--:|
| coarse STOP accuracy | 37.2 | 26.0 |
| FULL accuracy | 41.3 | 35.7 |
| best fixed (cross-fitted) | 40.2 | 34.9 |
| cross-replay oracle (cf) | 50.0 | 37.8 |
| **P_policy (headroom)** | **+9.8** [+5.1, +14.7] | **+2.8** [-2.2, +8.0] |
| H_net | +12.8 [+7.7, +17.9] | +3.1 [-2.0, +8.2] |
| any-action DMG | 13.8 | 7.7 |
| P(harm\|STOP✓): CROP | 46.4 | 43.3 |
| P(harm\|STOP✓): OCR | 10.7 | 40.0 |
| P(harm\|STOP✓): HIGHRES | 16.1 | 13.3 |
| STOP-optimal among decisive | 34.2 (n=79) | 25.4 (n=59) |

Best-action family distribution among decisive states (oracle picks):

| reader | STOP | CROP | OCR | HIGHRES |
|---|--:|--:|--:|--:|
| Qwen | 34.2 | 46.8 | 15.2 | 3.8 |
| Phi-4 | 25.4 | 50.8 | 22.0 | 1.7 |

## textvqa  (n = 200 / 200)

| metric | Qwen | Phi-4 |
|---|--:|--:|
| coarse STOP accuracy | 60.0 | 25.8 |
| FULL accuracy | 75.5 | 33.7 |
| best fixed (cross-fitted) | 69.1 | 33.3 |
| cross-replay oracle (cf) | 78.8 | 43.3 |
| **P_policy (headroom)** | **+9.8** [+4.7, +14.8] | **+10.0** [+6.0, +14.0] |
| H_net | -2.7 [-9.3, +4.0] | -0.3 [-7.3, +6.7] |
| any-action DMG | 13.0 | 7.0 |
| P(harm\|STOP✓): CROP | 16.1 | 26.2 |
| P(harm\|STOP✓): OCR | 10.2 | 9.5 |
| P(harm\|STOP✓): HIGHRES | 4.2 | 7.1 |
| STOP-optimal among decisive | 32.1 (n=81) | 23.0 (n=61) |

Best-action family distribution among decisive states (oracle picks):

| reader | STOP | CROP | OCR | HIGHRES |
|---|--:|--:|--:|--:|
| Qwen | 32.1 | 45.7 | 19.8 | 2.5 |
| Phi-4 | 23.0 | 32.8 | 37.7 | 6.6 |

Pooled (unweighted mean) P_policy: Qwen +9.8pp, Phi-4 +6.4pp.
