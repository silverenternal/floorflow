# Predicted-Control Room Identity Repair v1

## Decision

Two post-hoc identity routes are closed:

- component relabeling after a structural separation mask;
- background-query gating over frozen room-support fragments.

The active route is a zero-initialized dual-control residual that preserves the
native RGB path and can learn when predicted non-room layers are useful. It is
still a mechanism screen, not a promoted predicted-control result.

## Component Relabeling Failure

The structure-aware component classifier was audited on 16 validation cases.
Structure ink split native room support into `4-106` fragments per case and
removed `12-34%` of the support. Even an optimistic target-majority assignment
produced `3-75` room instances against `3-19` targets.

At the completed 256/64 screen:

| Arm | Room IoU | Instance F1 | Type accuracy | Count MAE |
|---|---:|---:|---:|---:|
| Frozen native support | 0.26795 | 0.17235 | 0.60365 | 3.406 |
| Component relabeling | 0.14936 | 0.09318 | 0.45781 | 14.891 |

Evidence:

- `outputs/train/native_component_semantic_assignfix256_v3/report.json`

## Query Readout Finding

The existing full-resolution query branch was not fully exploited. On
calibration64, blending query and dense probabilities improved all three
primary identity measures:

| Arm | Room IoU | Instance F1 | Count MAE |
|---|---:|---:|---:|
| Dense | 0.24685 | 0.21706 | 3.938 |
| Query blend 0.75 | 0.26195 | 0.24889 | 3.453 |

The offset-64 diagnostic agreed on direction. This supports continuing the
query path, but not claiming it as a new checkpoint before a disjoint holdout
and matched control.

Evidence:

- `outputs/reports/room_query_readout_audit_cal64_v1.json`
- `outputs/reports/room_query_readout_audit_holdout64_v1.json`

## Existence Gate Rejection

We added direct query-existence supervision, empty-mask supervision for
unmatched queries, and background-gated mask fusion. The 512/64, 16-epoch
screen did not promote:

- best selection score: `0.42839`;
- best Room IoU: `0.25192`;
- best instance F1: `0.25348`;
- best count MAE: `3.29688`.

Count calibration improved late, but Room IoU and instance F1 did not improve
together. The branch remains available for diagnostics and disabled by default.

Evidence:

- `outputs/train/room_query_existence_screen512_v1/report.json`

## Engineering Defects Fixed

The audit found four pipeline defects:

1. Warm-start state silently overwrote the explicitly selected room encoder.
   `--preserve-room-encoder` now keeps the room checkpoint authoritative.
2. Predicted furniture control took an `amax` over room, furniture, and
   footprint channels. It now reads only `[room_count:-1]`.
3. Predicted furniture saturated large interior regions and overwrote wall
   evidence. It is excluded from the control cache by default.
4. `render_room_masks` could bypass the model's fused readout. It now uses the
   explicit fused tensor when available.

## Predicted Control Cache

`outputs/cache/predicted_control_raster_v1` contains 512 training and 491
validation structural controls generated from predicted non-room layers.

- room labels in control: false;
- target used at inference: false;
- predicted furniture: excluded by default;
- room checkpoint and bridge SHA-256: recorded in each split manifest.

Visual inspection shows the current structure bridge preserves the exterior
well but misses many interior walls. Therefore predicted control is an
auxiliary residual, not a replacement for native RGB.

## Current Experimental Contract

The dual-control model receives six channels:

`native RGB (3) | predicted non-room control (3)`.

A `1x1` six-to-three residual projection is zero-initialized, so the disabled
path is exactly checkpoint-compatible with the native model. The current
512/64 screen must beat a compute-matched native-only arm on Room IoU, instance
F1, type accuracy, and count MAE, then repeat on an untouched offset-128
holdout. Failure returns the work to the detection-first instance decoder.

## Dual-Control Screen Result

The 512/64, 16-epoch dual-control screen did not pass:

| Arm | Room IoU | Instance F1 | Type accuracy | Count MAE |
|---|---:|---:|---:|---:|
| Native blend 0 | 0.18893 | 0.17009 | 0.48828 | 3.484 |
| Best dual blend 0.5 | 0.18781 | 0.18773 | 0.51432 | 3.750 |

The branch raised instance F1 but regressed Room IoU and count MAE, then degraded
further after epoch 3. It is not promoted. The projection remains disabled by
default and is retained only for controlled follow-up experiments with a
calibrated structural head.

Evidence:

- `outputs/train/room_query_dual_control_screen512_v2/report.json`

## Proposal Teacher Isolation

Class-agnostic Raster2Seq proposal pretraining was run for 24 epochs on
train256/cal64. Teacher-matched mask IoU rose from `0.2201` to roughly
`0.301-0.303`, so the support branch is learnable.

Two recovery schedules then failed the joint gate:

| Recovery | Room IoU | Instance F1 | Type accuracy | Count MAE |
|---|---:|---:|---:|---:|
| Native baseline | 0.18929 | 0.16523 | 0.47526 | 3.766 |
| GT fine-tune | 0.18158 | 0.19224 | 0.45052 | 3.219 |
| Frozen proposal, class-only | 0.18388 | 0.19388 | 0.50841 | 3.859 |

Proposal identity can improve, but the current fused label readout trades away
Room IoU/type or count. The branch is not promoted. The next model must preserve
the dense class map while predicting explicit instance boundaries.

Evidence:

- `outputs/train/room_query_teacher_only_256_24ep_v1/report.json`
- `outputs/train/room_query_teacher_pretrain_finetune256_v1/report.json`
- `outputs/train/room_query_teacher_classonly256_v1/report.json`

## Frozen-Feature Boundary Splitter

A 32-channel boundary head was trained for 16 epochs over frozen room-encoder
hidden features. The boundary target has about `2.1%` positive pixels.

The best calibrated readout at threshold `0.5` did not improve the baseline:

| Readout | Room IoU | Instance F1 | Type accuracy | Count MAE |
|---|---:|---:|---:|---:|
| Baseline | 0.18929 | 0.16523 | 0.47526 | 3.766 |
| Clear 0.3 | 0.156-0.167 | 0.142-0.157 | 0.472-0.514 | 3.92-4.13 |
| Clear 0.5 | 0.188-0.189 | 0.164-0.165 | 0.473-0.475 | 3.75-3.80 |
| Clear 0.7 | 0.18929 | 0.16523 | 0.47526 | 3.766 |

The frozen hidden field does not expose linearly separable instance
boundaries. The branch remains disabled. The next attempt must expose
multi-scale image features and train an image-side boundary decoder rather than
adding epochs to the same head.

Evidence:

- `outputs/train/room_boundary_splitter_screen512_v1/report.json`

## Multi-Scale and Dense-Conditioned Boundaries

Three follow-up boundary arms were screened under matched train256/cal64
budgets:

| Boundary input | Best selection | Boundary F1 | Result |
|---|---:|---:|---|
| Frozen multiscale skip | 0.28542 | 0.1506 at 0.3 | Reject |
| Independent native image | collapsed to empty | <0.16 | Reject |
| Dense posterior + native image | 0.28542 | <0.19 | Reject |

The target audit exposed an important issue: room count is already
over-predicted, so a split-only mechanism cannot close the identity gap. A
room-side seam target after removing exterior and different-class boundaries
has only about `0.0068%` positive pixels. All tested thresholds either made no
change, destroyed room support, or yielded low-precision boundaries.

Evidence:

- `outputs/train/room_boundary_multiscale_screen256_v1/report.json`
- `outputs/train/room_boundary_independent_seam_final_screen256_v1/report.json`
- `outputs/train/room_boundary_dense_seam_final_screen256_v1/report.json`

## Dense-Fragment Graph Arbitration

Calibration64 contains `11.31` predicted fragments against `9.27` target room
instances. Only `5.25` fragments reach IoU >= 0.2 with any target; `12.95`
fragment pairs point to different targets while only `0.59` pairs point to the
same target. This supports joint existence and merge arbitration, but also
shows that most predicted fragments are spurious rather than duplicate.

| Arm | Room IoU | Instance F1 | Type accuracy | Count MAE |
|---|---:|---:|---:|---:|
| Dense baseline | 0.18929 | 0.16523 | 0.47526 | 3.766 |
| Guarded graph, best joint point | 0.18952 | 0.16666 | 0.48307 | 3.750 |
| Guarded graph, best composite | 0.18902 | 0.17013 | 0.48307 | 3.766 |
| Node target IoU 0.5 | 0.18931 | 0.16523 | 0.47526 | 3.766 |

The preserved dense map is not a reliable graph substrate: stronger arbitration
improves instance metrics by deleting/relabelling semantically important
fragments. With IoU-0.5 node labels, edge positives vanish and the readout
becomes a no-op. The mechanism is retained, but dense fragments are rejected
for promotion.

Evidence:

- `outputs/train/room_fragment_graph_guarded_screen256_v1/report.json`
- `outputs/train/room_fragment_graph_iou50_screen256_v1/report.json`

## Query-Mask Instance Graph

Query masks are a better node substrate than dense fragments:

- `8.89` valid masks per case;
- `6.03` masks reach IoU >= 0.2 with a target;
- `2.22` query pairs point to the same target;
- query-only Room IoU `0.21560`, instance F1 `0.22245`, count MAE
  `3.34375`.

The existing query existence head is not calibrated: thresholds 0.2, 0.4, and
0.6 produce identical predictions. Learned query graph arbitration did not beat
the query-only baseline:

| Arm | Room IoU | Instance F1 | Count MAE |
|---|---:|---:|---:|
| Query only | 0.21560 | 0.22245 | 3.344 |
| Best learned graph | 0.21557 | 0.21981 | 3.359 |

Class-safe duplicate merge reduces instance F1 because learned edge precision
remains below the precision needed for a safe merge. Query support IoU is only
`0.5593`, so the next bottleneck is recovering missing room support without
destroying query identity.

Evidence:

- `outputs/train/query_mask_graph_existence_screen256_v1/report.json`

## Dense-Query Support Gate

The learned residual gate initialized as fixed blend75. It did not beat the
fixed blend: after 16 epochs the learned gate reached Room IoU `0.25942`,
instance F1 `0.24618`, and count MAE `3.719`, while fixed blend75 reached
`0.26195`, `0.24889`, and `3.453`.

The fixed blend50 readout is the strongest candidate. Full val491 paired
bootstrap:

| Metric | Dense | Blend50 | Paired result |
|---|---:|---:|---|
| Room IoU | 0.32358 | 0.32855 | +0.00497, CI [0.00226, 0.00780] |
| Instance F1 | 0.25830 | 0.27125 | +0.01295, CI [0.00443, 0.02101] |
| Count MAE | 4.242 | 3.988 | -0.25458, CI [-0.45010, -0.05703] |
| Type accuracy | 0.67437 | 0.68237 | +0.00799, CI crosses zero |
| Room support IoU | 0.77530 | 0.77530 | exact tie |

Offset-128 holdout64 agrees in direction for Room IoU, instance F1, and count,
but all confidence intervals cross zero and type accuracy is lower. The
blend50 branch is therefore promoted to five-seed validation, not to a final
claim.

Evidence:

- `outputs/train/dense_query_support_gate_blend75_screen256_v1/report.json`
- `outputs/reports/dense_query_blend_val491_stats_v1.json`
- `outputs/reports/dense_query_blend_holdout128_v1.json`
