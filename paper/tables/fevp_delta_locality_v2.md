# FEVP Delta and Locality Distributions

Status: aggregate FEVP distribution evidence for its declared row set. The
submission-facing per-operation main result is
`residential_sota_main_v5.md`.

| Metric | Mean | P10 | Median | P90 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Delta-IoU | 0.987084 | 0.990585 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| Delta-MSE | 0.000218 | 0.000000 | 0.000000 | 0.000189 | 0.000000 | 0.041115 |
| Locality violation | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

| Arm | Violation mean | Violation P90 | Delta-IoU mean |
|---|---:|---:|---:|
| dense_cross_attention | 0.000455107 | 0.000227324 | 0.887741 |
| no_edit_gating | 0.013884758 | 0.009716790 | 0.853715 |
| no_structural_graph | 0.000000000 | 0.000000000 | 0.888817 |
| typed_edit_graph | 0.000000000 | 0.000000000 | 0.937473 |

Operation, room-count, edit-area, and changed-room strata are in the JSON report.
