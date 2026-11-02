# Main Predicted-Control Reliability v1

Status: frozen scoped release. Panel A uses CubiCasa val491 with five
repeated target-free seeds. Panel B uses 738 development-exposed test503
prompts. Semantic realization is a diagnostic from an older val191 model,
not the selected release checkpoint.

## Panel A: Structural reasonableness (val491, five seeds)

| Metric | Mean |
|---|---:|
| Raster plausible | 0.984929 |
| Plan plausible | 0.986558 |
| Room coverage | 1.000000 |
| Room-face retention | 0.995655 |
| FP-Eval validity | 1.000000 |
| Raw watertight | 0.816701 |
| Functional watertight | 0.997963 |
| Wall continuity | 0.995427 |
| Opening-wall align | 0.993530 |
| Furniture containment | 1.000000 |
| Outside-footprint fraction | 0.000000 |
| Room overlap ratio | 0.000000 |
| Room macro IoU (diag.) | 0.269701 |

Raw watertightness is `0.81670`, not `1.0`. Of 491 footprints, 401 are
single-component and pass both raw and componentwise checks; 90 are
multi-component and exceed the standard single-region FP-Eval boundary.
Room macro IoU is a same-pipeline fidelity diagnostic and does not gate
the reasonableness claim.

## Panel B: Execution safety and semantic boundary

| Evidence | Observed |
|---|---:|
| Supported prompts executed | 438 / 438 |
| Unsupported/ambiguous abstained | 300 / 300 |
| Outside pixel exact (minimum) | 1.000000 |
| Outside room-label exact (minimum) | 1.000000 |
| Validity (minimum) | 1.000000 |

| Semantic diagnostic | Successes / prompts | Rate |
|---|---:|---:|
| Room add | 92 / 100 | 0.9200 |
| Room type | 55 / 100 | 0.5500 |
| Room remove | 20 / 100 | 0.2000 |
| Opening delete | 30 / 59 | 0.5085 |
| Furniture delete | 0 / 15 | 0.0000 |

Object-delete and ghost-state audits pass their deterministic contracts;
they are not semantic-success rows. No fresh held-out claim is made.

<!-- registry bindings: predicted.val491.raster_plausible, predicted.val491.plan_plausible, predicted.val491.room_coverage, predicted.val491.room_face_retention, predicted.val491.validity, predicted.val491.watertight, predicted.val491.functional_watertight, predicted.val491.wall_continuity_rate, predicted.val491.opening_wall_alignment, predicted.val491.furniture_room_containment, predicted.val491.outside_footprint_fraction, predicted.val491.room_overlap_ratio, predicted.val491.room_macro_iou_diagnostic, operation.safety.prompts, operation.safety.executed, operation.safety.abstained -->
