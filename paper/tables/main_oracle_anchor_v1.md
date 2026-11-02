# Main Oracle-Control Editing Anchor v1

Status: frozen scoped release. CubiCasa5K test503 declared oracle controls;
2,479 paired edits over five repeated execution seeds. Intervals resample
floorplans, not duplicated seed rows.

| Operation | N | Delta-IoU [95% CI] | Delta-MSE | Outside exact | Vector validity |
|---|---:|---:|---:|---:|---:|
| Room type | 487 | 0.999199 [0.998420, 0.999747] | 0.000069 | 1.000000 | 0.956879 |
| Room add | 495 | 0.999801 [0.999545, 0.999990] | 0.000030 | 1.000000 | 0.955556 |
| Room remove | 500 | 0.999756 [0.999479, 0.999966] | 0.000019 | 1.000000 | 0.954000 |
| Opening delete | 502 | 0.956703 [0.941316, 0.970424] | 0.000096 | 1.000000 | 0.912351 |
| Furniture delete | 495 | 0.987812 [0.982565, 0.992375] | 0.000083 | 1.000000 | 0.951515 |

Pooled FP-Eval validity is `1.0`; conditional watertightness is
`0.957875-0.967491`. These are pooled five-operation diagnostics, not
operation-specific estimates. FEVP five-check validity is `0.9124-0.9569`.
Delta-IoU and Delta-MSE measure binary change support, not requested-class
correctness or reference reconstruction fidelity.

<!-- registry bindings: oracle.room_type.cases, oracle.room_type.delta_iou, oracle.room_type.delta_mse, oracle.room_type.outside_exact_ratio, oracle.room_add.cases, oracle.room_add.delta_iou, oracle.room_add.delta_mse, oracle.room_add.outside_exact_ratio, oracle.room_remove.cases, oracle.room_remove.delta_iou, oracle.room_remove.delta_mse, oracle.room_remove.outside_exact_ratio, oracle.opening_delete.cases, oracle.opening_delete.delta_iou, oracle.opening_delete.delta_mse, oracle.opening_delete.outside_exact_ratio, oracle.furniture_delete.cases, oracle.furniture_delete.delta_iou, oracle.furniture_delete.delta_mse, oracle.furniture_delete.outside_exact_ratio -->
