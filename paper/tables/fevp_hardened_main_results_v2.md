# Hardened FEVP Main Results v2

Protocol: FloorPlan Editing Verification Protocol 1.0-hardened.

Status: this is the aggregate FEVP hardening table over its declared row set.
The current submission headline is the per-operation
`residential_sota_main_v5.md`; do not substitute this aggregate range for the
per-operation result.

## Metric Contract

| Metric | Reference implementation | Fixed tolerance | Negative controls |
|---|---|---|---|
| outside_exact | `floorflow.evaluation.fevp_metric_contracts` | room_match_min_iou=0.5, room_polygon_hausdorff_px=0.5, room_min_area_px=4.0, room_overlap_area_px2=4.0, edge_contact_px=2.0, edge_snap_px=1.5, wall_junction_cluster_px=2.0, hole_min_area_px=0.25, changed_pixel_epsilon=0.0, float_report_decimals=6 | 3 |
| validity | `floorflow.evaluation.fevp_metric_contracts` | room_match_min_iou=0.5, room_polygon_hausdorff_px=0.5, room_min_area_px=4.0, room_overlap_area_px2=4.0, edge_contact_px=2.0, edge_snap_px=1.5, wall_junction_cluster_px=2.0, hole_min_area_px=0.25, changed_pixel_epsilon=0.0, float_report_decimals=6 | 5 |
| conditional_watertightness | `floorflow.evaluation.fevp_metric_contracts` | room_match_min_iou=0.5, room_polygon_hausdorff_px=0.5, room_min_area_px=4.0, room_overlap_area_px2=4.0, edge_contact_px=2.0, edge_snap_px=1.5, wall_junction_cluster_px=2.0, hole_min_area_px=0.25, changed_pixel_epsilon=0.0, float_report_decimals=6 | 3 |
| node_f1 | `floorflow.evaluation.fevp_metric_contracts` | room_match_min_iou=0.5, room_polygon_hausdorff_px=0.5, room_min_area_px=4.0, room_overlap_area_px2=4.0, edge_contact_px=2.0, edge_snap_px=1.5, wall_junction_cluster_px=2.0, hole_min_area_px=0.25, changed_pixel_epsilon=0.0, float_report_decimals=6 | 3 |
| edge_overlap | `floorflow.evaluation.fevp_metric_contracts` | room_match_min_iou=0.5, room_polygon_hausdorff_px=0.5, room_min_area_px=4.0, room_overlap_area_px2=4.0, edge_contact_px=2.0, edge_snap_px=1.5, wall_junction_cluster_px=2.0, hole_min_area_px=0.25, changed_pixel_epsilon=0.0, float_report_decimals=6 | 3 |

## Editing Quality

| Metric | Mean | SD | 95% CI | Worst seed |
|---|---:|---:|---:|---:|
| delta_iou | 0.987084 | 0.002096 | [0.984482, 0.989686] | 22 (0.984774) |
| delta_mse | 0.000218 | 0.000037 | [0.000171, 0.000264] | 20 (0.000259) |
| ged | 0.869544 | 0.065263 | [0.788510, 0.950579] | 21 (0.983461) |

| GED comparison | Mean paired delta | Max seed p (one-sided) |
|---|---:|---:|
| Typed - dense_cross_attention | -0.064623 | 2.834e-09 |
| Typed - no_edit_gating | -0.327793 | 1.489e-27 |
| Typed - no_structural_graph | -0.061396 | 7.268e-09 |

## Strict Locality

| Arm | Pixel exact | Room exact | Label exact | Polygon exact | Edge exact | Violation rate |
|---|---:|---:|---:|---:|---:|---:|
| Strict | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000000 |
| No gate | 0.986115 | 0.486325 | 0.945139 | 0.488503 | 0.953046 | 0.013884758 |

Single-room negative control: 7345 cases; strict room-level exactness 1.000000.

## Topology Legality

| Check | Pass rate |
|---|---:|
| validity | 0.979750 |
| outer_boundary_closed | 1.000000 |
| room_polygon_closed | 1.000000 |
| no_self_intersection | 1.000000 |
| no_room_overlap | 0.980557 |
| semantic_complete | 0.999193 |

| Watertight metric | Raw | Guarded |
|---|---:|---:|
| conditional_watertightness_fpeval | 0.963553 | 0.962637 |
| watertight_rate | 0.382977 | 0.841146 |
| hole_count | 0.876967 | 0.143122 |
| hole_area_ratio | 0.012983 | 0.001925 |
| disconnected_component_count | 0.241388 | 0.083179 |
| dangling_endpoint_rate | 0.094046 | 0.021242 |

Guard repairs: 57; regressions: 67; repair rate among failures: 0.143216.
The guard is not presented as a net-benefit contribution.

## Graph Consistency

| View | Node P | Node R | Node F1 | Edge P | Edge R | Edge F1/overlap | GED |
|---|---:|---:|---:|---:|---:|---:|---:|
| Primary full method | 0.999361 | 0.969127 | 0.982250 | n/a | n/a | 0.929475 | 0.869544 |
| Typed routing mechanism | 0.995885 | 0.925830 | 0.955068 | 0.910891 | 0.823376 | 0.890007 | 2.120855 |

## Protocol Attacks

| Mandatory metric | Attacks | Sensitive | False-pass rate |
|---|---:|---:|---:|
| outside_exact | 4 | 4 | 0.000 |
| validity | 6 | 6 | 0.000 |
| conditional_watertightness | 4 | 4 | 0.000 |
| node_f1 | 6 | 6 | 0.000 |
| edge_overlap | 5 | 5 | 0.000 |

Hardened validity is a five-check conjunction and is not the old single fpeval flag. The watertight guard is reported as a failed net-benefit mechanism whenever regressions exceed repairs.
