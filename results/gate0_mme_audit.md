# Gate 1 Audit Table

**Verdict: CONDITIONAL** (pre-registered thresholds, SPEC §9)  
H_net = 12.8pp (CI low 7.7), P_policy = 9.8pp (CI low 5.1), DMG = 13.8%, diverse datasets 1/1

## Table 1 — accuracy and headroom (%)

| set | n | coarse STOP | full image | best fixed (cf) | naive oracle | oracle (cf) | dummy oracle | H net | P policy |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| pooled | 196 | 37.2 | 41.3 | 40.2 | 64.8 | 50.0 | 35.7 | 12.8 | 9.8 |
| mme_realworld | 196 | 37.2 | 41.3 | 40.2 | 64.8 | 50.0 | 35.7 | 12.8 | 9.8 |

## Table 2 — best action type on decisive samples (%)

| set | n decisive | STOP | CROP | OCR | HIGHRES | max share |
|---|--:|--:|--:|--:|--:|--:|
| pooled | 79 | 34.2 | 46.8 | 15.2 | 3.8 | 46.8 |
| mme_realworld | 79 | 34.2 | 46.8 | 15.2 | 3.8 | 46.8 |

## Table 3 — exclusive solve and harm (%)

| set | excl STOP | excl CROP | excl OCR | excl HIGHRES | DMG | harm CROP | harm OCR | harm HIGHRES |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| pooled | 0.0 | 12.2 | 0.5 | 1.5 | 13.8 | 46.4 | 10.7 | 16.1 |
| mme_realworld | 0.0 | 12.2 | 0.5 | 1.5 | 13.8 | 46.4 | 10.7 | 16.1 |

## Table 4 — recovery overlap among base failures (counts)

| recovering action types | samples |
|---|--:|
| none | 88 |
| CROP | 24 |
| CROP+HIGHRES | 13 |
| CROP+HIGHRES+OCR | 6 |
| CROP+OCR | 4 |
| HIGHRES | 3 |
| OCR | 1 |
| HIGHRES+OCR | 1 |

## Table 5 — budget-bottleneck subset (coarse wrong AND full-image right)

n = 28 (14.3% of samples); recovered by at least one action: 92.9%

| | STOP | CROP | OCR | HIGHRES |
|---|--:|--:|--:|--:|
| solves the sample | 0.0 | 78.6 | 21.4 | 71.4 |
| is the best action | 0.0 | 65.4 | 23.1 | 11.5 |
| solves it exclusively | 0.0 | 21.4 | 0.0 | 10.7 |

recovery overlap inside the subset: CROP+HIGHRES=11, CROP=6, CROP+HIGHRES+OCR=5, HIGHRES=3, none=2, HIGHRES+OCR=1

## Table 6 — label reliability

- per-action R1/R2 agreement: 88.0%
- top-1 family agreement (decisive): 70.9%
- parse failures: 0 / 7000 rows


## Breakdown by subtask

| subtask | n | coarse | full | oracle cf | dummy | H net | best STOP | CROP | OCR | HIGHRES |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Autonomous_Driving | 79 | 26.6 | 29.1 | 36.7 | 30.4 | 6.3 | 32.0 | 64.0 | 4.0 | 0.0 |
| Diagram and Table | 12 | 58.3 | 50.0 | 75.0 | 71.4 | 14.3 | 66.7 | 0.0 | 22.2 | 11.1 |
| Monitoring | 54 | 37.0 | 27.8 | 48.1 | 50.0 | 16.7 | 42.1 | 52.6 | 0.0 | 5.3 |
| OCR with Complex Context | 36 | 61.1 | 88.9 | 75.0 | — | 13.9 | 21.1 | 31.6 | 42.1 | 5.3 |
| Remote Sensing | 15 | 20.0 | 33.3 | 46.7 | — | 26.7 | 14.3 | 71.4 | 14.3 | 0.0 |


## Breakdown by category

| category | n | coarse | full | oracle cf | dummy | H net | best STOP | CROP | OCR | HIGHRES |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Attention_TrafficSignal | 6 | 33.3 | 33.3 | 33.3 | 33.3 | 0.0 | 66.7 | 33.3 | 0.0 | 0.0 |
| Attribute_Motion_MultiPedestrians | 6 | 16.7 | 0.0 | 16.7 | 16.7 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| Attribute_Motion_MultiVehicles | 6 | 16.7 | 16.7 | 33.3 | 16.7 | 16.7 | 0.0 | 100.0 | 0.0 | 0.0 |
| Attribute_Motion_Pedestrain | 6 | 50.0 | 16.7 | 50.0 | 50.0 | 0.0 | 50.0 | 50.0 | 0.0 | 0.0 |
| Attribute_Motion_Vehicle | 6 | 16.7 | 16.7 | 33.3 | 16.7 | 16.7 | 50.0 | 50.0 | 0.0 | 0.0 |
| Attribute_Visual_TrafficSignal | 5 | 40.0 | 40.0 | 40.0 | 40.0 | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| Character Identification | 6 | 50.0 | 83.3 | 83.3 | — | 33.3 | 25.0 | 25.0 | 50.0 | 0.0 |
| Object_Count | 4 | 50.0 | 100.0 | 75.0 | 75.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| Objects_Identify | 5 | 20.0 | 40.0 | 60.0 | 80.0 | -20.0 | 33.3 | 66.7 | 0.0 | 0.0 |
| Person/counting | 1 | (too small) | | | | | | | | |
| Prediction_Intention_Ego | 5 | 0.0 | 20.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| Prediction_Intention_Pedestrian | 5 | 40.0 | 40.0 | 40.0 | 20.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| Prediction_Intention_Vehicle | 5 | 40.0 | 60.0 | 60.0 | 20.0 | 20.0 | 66.7 | 33.3 | 0.0 | 0.0 |
| Relation_Interaction_Ego2Pedestrain | 5 | 20.0 | 20.0 | 20.0 | 20.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Relation_Interaction_Ego2TrafficSignal | 5 | 20.0 | 20.0 | 40.0 | 20.0 | 20.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| Relation_Interaction_Ego2Vehicle | 5 | 40.0 | 40.0 | 40.0 | 40.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| Relation_Interaction_Other2Other | 5 | 0.0 | 0.0 | 20.0 | 20.0 | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| Scene Understanding | 5 | 100.0 | 80.0 | 100.0 | — | 0.0 | 50.0 | 0.0 | 25.0 | 25.0 |
| Vehicle/counting | 5 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| adver_and_product | 5 | 40.0 | 100.0 | 60.0 | — | 20.0 | 25.0 | 25.0 | 50.0 | 0.0 |
| book_map_poster | 5 | 60.0 | 60.0 | 60.0 | — | 0.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| calculate | 5 | 20.0 | 0.0 | 20.0 | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| color | 5 | 20.0 | 60.0 | 80.0 | — | 60.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| count | 5 | 0.0 | 0.0 | 0.0 | — | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| diagram | 6 | 83.3 | 66.7 | 100.0 | 100.0 | 0.0 | 83.3 | 0.0 | 16.7 | 0.0 |
| intention | 4 | 0.0 | 0.0 | 50.0 | — | 50.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| license | 5 | 60.0 | 100.0 | 60.0 | — | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| person/attribute/color | 5 | 0.0 | 0.0 | 20.0 | 0.0 | 20.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| person/attribute/orientation | 4 | 0.0 | 0.0 | 25.0 | 0.0 | 25.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| person/counting | 5 | 100.0 | 80.0 | 80.0 | — | -20.0 | 50.0 | 50.0 | 0.0 | 0.0 |
| phone_and_address | 5 | 40.0 | 100.0 | 60.0 | — | 20.0 | 0.0 | 50.0 | 50.0 | 0.0 |
| position | 5 | 40.0 | 40.0 | 60.0 | — | 20.0 | 33.3 | 33.3 | 33.3 | 0.0 |
| property | 5 | 20.0 | 20.0 | 40.0 | — | 20.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| table | 6 | 33.3 | 33.3 | 50.0 | 50.0 | 25.0 | 33.3 | 0.0 | 33.3 | 33.3 |
| text_recog | 5 | 80.0 | 100.0 | 100.0 | — | 20.0 | 0.0 | 100.0 | 0.0 | 0.0 |
| vehicle/attribute/color | 5 | 20.0 | 20.0 | 40.0 | — | 20.0 | 50.0 | 50.0 | 0.0 | 0.0 |
| vehicle/attribute/orientation | 5 | 60.0 | 0.0 | 60.0 | — | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| vehicle/counting | 5 | 20.0 | 20.0 | 40.0 | — | 20.0 | 50.0 | 50.0 | 0.0 | 0.0 |
| vehicle/location | 5 | 40.0 | 40.0 | 40.0 | — | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |
