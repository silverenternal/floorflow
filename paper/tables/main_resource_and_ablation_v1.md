# Main Matched Controls and Composition Ablation v1

Status: frozen scoped release. Same-case oracle-control comparisons; the
Raster2Seq row is an outside-frozen redraw wrapper and the editing-aware
control is intentionally weak. Neither is a native strong editing model.

| Operation | N | FloorFlow | Raster2Seq wrapper | Editing-aware | Paired gain vs Raster2Seq | Paired gain vs editing-aware |
|---|---:|---:|---:|---:|---:|---:|
| Room type | 487 | 0.999199 | 0.137160 | 0.909800 | 0.862039 | 0.089486 |
| Room add | 495 | 0.999801 | 0.166756 | 0.995985 | 0.833044 | 0.004015 |
| Room remove | 500 | 0.999756 | 0.177266 | 0.998508 | 0.822490 | 0.001340 |
| Opening delete | 502 | 0.956703 | 0.189491 | 0.731151 | 0.767212 | 0.225552 |
| Furniture delete | 495 | 0.987812 | 0.145367 | 0.742469 | 0.842445 | 0.245343 |

The frozen Raster2Seq wrapper has outside exactness `1.0`; this is a
wrapper property, not native editing awareness. Paired confidence
intervals use one difference per floorplan. The editing-aware runner
recomputes its own paired FloorFlow row, so its absolute values can differ
slightly from T1; comparisons stay within their own run.

## Composition-policy ablation

| Arm | Outside pixel | Outside room | Violation | Delta-IoU | Node F1 | Edge F1 | GED |
|---|---:|---:|---:|---:|---:|---:|---:|
| Typed routing | 1.000000 | 1.000000 | 0.000000 | 0.937473 | 0.955068 | 0.890007 | 2.120855 |
| Dense routing | 0.999545 | 0.999470 | 0.000455 | 0.887741 | 0.953797 | 0.887080 | 2.185478 |
| No edit gating | 0.986115 | 0.983599 | 0.013885 | 0.853715 | 0.949057 | 0.874121 | 2.448649 |
| No structural graph | 1.000000 | 1.000000 | 0.000000 | 0.888817 | 0.953819 | 0.887200 | 2.182251 |

Arms are deterministic composition policies, not separately trained
networks. At matched observable routing budget, typed routing adds
Delta-IoU `+0.048656` and changes GED by `-0.061396` with zero
added learned parameters. It consumes more routing time
(`3.1925` vs `0.2666` ms mean); efficiency is
not claimed. All 26 deterministic protocol attacks are sensitive.

<!-- registry bindings: oracle.room_type.matched_editing_aware_iou, oracle.room_add.matched_editing_aware_iou, oracle.room_remove.matched_editing_aware_iou, oracle.opening_delete.matched_editing_aware_iou, oracle.furniture_delete.matched_editing_aware_iou, oracle.room_type.frozen_raster2seq_delta_iou_mean, oracle.room_add.frozen_raster2seq_delta_iou_mean, oracle.room_remove.frozen_raster2seq_delta_iou_mean, oracle.opening_delete.frozen_raster2seq_delta_iou_mean, oracle.furniture_delete.frozen_raster2seq_delta_iou_mean, mechanism.typed_routing, mechanism.dense_cross_attention, mechanism.no_edit_gating, mechanism.no_structural_graph, mechanism.budget.typed_latency_ms, mechanism.budget.no_graph_latency_ms, mechanism.budget.delta_iou_gain, mechanism.budget.ged_gain, robustness.attack_suite -->
