# Predicted-Control Fidelity Baseline v3

## Scope

This is the first target-comparative baseline for `RC-03B`. It uses the frozen
five-seed-safe output representation:

- native CubiCasa raster input;
- the existing ResNet18-FPN room-control model;
- structure-only bridge seed19;
- explicit room-cell polygons;
- union with one-pixel room seams;
- 0.04-frame endpoint gap reconciliation;
- no target information at inference.

It evaluates all 491 validation floors and leaves test503 closed.

## Canonical Evaluator

`src/floorflow/evaluation/predicted_control_fidelity.py` computes all metrics
from one explicit-room prediction plan:

- raster Room macro IoU and room-support IoU;
- matched instance geometry, instance F1, and type accuracy;
- fpeval wall/opening/Room F1, label accuracy, and edit cost;
- HouseMind-style Node F1, edge overlap, and fixed GED;
- Raster2Seq-style explicit-polygon corner and angle F1;
- the frozen raster and CAD plausibility suite.

GED uses the fixed NetworkX evaluator with a 0.1-second per-case bound. Every row
retains the `exact` indicator; timed-out cases are not reported as exact GED.

## Results

| Metric | Val491 |
|---|---:|
| Room macro IoU | 0.26972 |
| Room support IoU | 0.77268 |
| Matched geometric IoU | 0.55315 |
| Instance F1 @0.5 | 0.23065 |
| Instance type accuracy | 0.58825 |
| Class-constrained instance F1 | 0.17091 |
| Predicted / target room count | 9.560 / 8.444 |
| Mean room-count absolute error | 3.2179 |
| fpeval wall F1 / precision / recall | 0.4419 / 0.3600 / 0.6579 |
| fpeval opening F1 | 0.08295 |
| fpeval Room F1 | 0.36116 |
| fpeval Room mean IoU on matches | 0.64212 |
| fpeval Room label accuracy | 0.65544 |
| Node F1 | 0.66020 |
| Edge F1 / overlap | 0.37724 / 0.26462 |
| GED / exact rate | 11.6008 / 0.6884 |
| Corner F1 / Angle F1 | 0.38069 / 0.60554 |
| Edit cost per GT wall | 12.9597 |

Structural gates remain intact: validity `1.0`, room-face retention `0.9957`,
functional watertightness `0.9980`, wall continuity `0.9955`, and plan
plausibility `0.9878`.

## Diagnosis

The bottleneck is not room support or CAD legality. Room support IoU is `0.773`
and matched-room mean geometric IoU is `0.553`. The failure is room identity:

- predicted room count exceeds target by `1.12` on average;
- only `1.93` instances per case survive geometric matching at IoU 0.5;
- type accuracy on matched instances is `0.588`;
- class-constrained instance F1 is only `0.171`.

Wall geometry is over-produced: recall `0.658` but precision `0.360`.
Opening recognition is the weakest physical-layer metric at F1 `0.083`.

The graph gap is also identity-dominated: Node F1 `0.660`, edge overlap `0.265`,
and GED `11.60`. Pixel-space auxiliary losses alone are unlikely to solve this;
the next representation must supervise room-existence queries, semantic
identity, and graph assignment directly.

## Checkpoint Screen

The deployable native-room checkpoint remains
`outputs/train/raster_room_control_native_full_v1/best.pt`.

The higher `0.375` Room IoU from `raster_room_control_full_v1` uses
non-room-raster control rendered from target structure. It is
reconstruction-only and is excluded from the predicted-control row.

The existing native instance-query screen remains below the native dense model
and does not justify promotion without a new identity objective.

Official Raster2Seq reports much higher val Room IoU (`~0.65`), but it is a
separate reconstruction threat. A FloorFlow hybrid using Raster2Seq proposals
must be labeled as a hybrid and cannot be presented as standalone FloorFlow.

## Next Action

Build one identity-aware room-control screen on train256/val64:

- preserve the native raster encoder;
- add explicit room-existence and room-type query supervision;
- use Hungarian mask/class assignment against GT room instances;
- keep the explicit-room + gap-reconciled output contract;
- reject the arm unless Room IoU, instance F1, type accuracy, Node F1, and edge
  overlap improve on both calibration64 and disjoint holdout64 without losing
  the five-seed structural gates.

Machine-readable evidence:

- `outputs/reports/predicted_control_fidelity_val491_v3.json`

## Identity Calibration Screens

Two low-risk post-hoc routes were tested on calibration64 before any new
end-to-end training.

### Oracle count merge

Using the target room count only as a diagnostic ceiling, fragment merging
reduced count absolute error from `3.266` to `1.094` and raised instance F1 from
`0.172` to `0.202`. Room macro IoU fell from `0.269` to `0.244`, however,
because the deterministic merge policy joined fragments across incorrect room
boundaries. A better count predictor alone is therefore insufficient.

A stricter adjacency-only merge reduced count error to `1.453` but further
lowered Room IoU to `0.169` and type accuracy to `0.333`. Count
post-processing is rejected in both forms.

### Image-aware component classifier

A classifier over frozen room support used current class probabilities, native
RGB statistics, neighboring class probabilities, and component shape features.
Across 14 training epochs its best component accuracy was `0.493`, while
end-to-end calibration Room IoU never exceeded `0.045` versus the `0.268`
baseline. Relabeling adjacent components to the same class merged distinct
instances and increased count error to `7.094`.

Both post-hoc semantic calibration routes are rejected. The next candidate must
jointly predict room existence, support, and identity rather than reclassifying
frozen masks.

Additional evidence:

- `outputs/reports/room_identity_diagnostic_val64_v4.json`
- `outputs/train/native_component_semantic_screen256_v1/report.json`
