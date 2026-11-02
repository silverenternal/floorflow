# Residential End-to-End Five-Seed Study

| Seed | Refiner gate | Room IoU | Room IoU gain | Validity | Watertight |
|---|---|---:|---:|---:|---:|
| 19 | passed | 0.071583 | +0.029335 | 1.000000 | 0.391039 |
| 20 | passed | 0.071530 | +0.029282 | 1.000000 | 0.553971 |
| 21 | passed | 0.072141 | +0.029894 | 1.000000 | 0.409369 |
| 22 | passed | 0.074032 | +0.031785 | 1.000000 | 0.376782 |
| 23 | passed | 0.070480 | +0.028233 | 1.000000 | 0.574338 |

| Endpoint | Delta | Wilcoxon p | Holm p | Rank-biserial |
|---|---:|---:|---:|---:|
| graph_node_f1 | +0.309065 | 1.082e-128 | 7.572e-128 | +0.9203 |
| graph_edge_overlap | +0.217459 | 6.545e-78 | 1.527e-77 | +0.7568 |
| graph_ged | -12.149682 | 1.271e-127 | 4.450e-127 | -0.9276 |
| external_room_iou | +0.046549 | 2.751e-46 | 4.815e-46 | +0.7499 |
| external_node_f1 | +0.198433 | 2.957e-43 | 4.139e-43 | +0.7395 |
| external_edge_overlap | +0.066381 | 1.978e-28 | 2.308e-28 | +0.6353 |
| external_ged | -1.356415 | 1.234e-09 | 1.234e-09 | -0.3216 |

Interaction promotion: **rejected_unstable**.

Refiner latency: 0.882 ms/case; graph readout: 86.527 ms/case; extra neural NFE: 1.
