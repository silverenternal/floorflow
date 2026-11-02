# Dense-Query Blend Five-Seed Audit v1

## Scope

This audit evaluates the fixed probability blend

`0.5 * dense_room_probability + 0.5 * query_fused_probability`

under the predicted-control CubiCasa protocol. The dense and query branches are
independent checkpoints. Inference never reads the target. CubiCasa test remains
closed.

Training uses 256 train cases, 64 calibration cases, 384 pixels, batch 8, and
eight epochs for seeds 19-23. The room encoder is frozen and the query decoder
is initialized from the existing 384px promotion checkpoint.

## Full Validation

All five seeds improve Room IoU, instance F1, and count MAE over the matched
dense checkpoint on val491. Support IoU is an exact tie.

| Metric | Dense | Blend range | Mean delta | SD | Improved seeds |
|---|---:|---:|---:|---:|---:|
| Room IoU | 0.32358 | 0.32565-0.32838 | +0.00287 | 0.00111 | 5/5 |
| Instance F1 | 0.25830 | 0.26578-0.27078 | +0.01009 | 0.00178 | 5/5 |
| Count MAE | 4.24236 | 3.87780-4.05295 | -0.28961 | 0.06368 | 5/5 |
| Type accuracy | 0.67437 | 0.66953-0.68406 | +0.00172 | 0.00539 | 3/5 |
| Room support IoU | 0.77530 | 0.77530 | 0.00000 | 0.00000 | tie |

Instance F1 has a paired 95% interval above zero on all five seeds. Count MAE
has an interval below zero on four of five seeds. Room IoU is directionally
positive on all seeds, but only one seed has an interval excluding zero.
Type accuracy is not promoted because its direction changes across seeds.

## Stability Boundary

Calibration64 improves Room IoU and instance F1 on all five seeds. The
offset-128 holdout does not:

- Room IoU improves on 5/5 seeds with mean delta `+0.00227`.
- Count MAE improves on 4/5 seeds with mean delta `-0.17813`.
- Instance F1 improves on only 1/5 seeds with mean delta `-0.00362`.

The candidate therefore does not pass the predeclared all-metric stability
gate. Full-val gains cannot hide the holdout disagreement, and seeds must not be
pooled to manufacture significance.

## Structural Consumer

The candidate is consumed by the complete predicted-CAD bridge at its trained
224px room-control resolution. The same seed's bridge is paired with its query
checkpoint. All five val491 structure reports pass validity, functional-
watertightness, and outside-footprint gates.

| Structure metric | Mean paired delta | Improved seeds | CI excludes zero |
|---|---:|---:|---:|
| Room macro IoU | +0.00337 | 5/5 | 3/5 |
| Room support IoU | +0.00074 | 5/5 | 5/5 |
| Instance F1 | +0.00607 | 5/5 | 1/5 |
| Count MAE | -0.21426 | 5/5 | 3/5 |
| Wall F1 | +0.00754 | 5/5 | 5/5 |
| Opening F1 | -0.00475 | 0/5 | 2/5 |
| Node F1 | +0.01488 | 5/5 | 4/5 |
| Edge overlap | +0.01462 | 5/5 | 4/5 |
| GED | -1.03707 | 5/5 | 5/5 |
| Wall continuity | +0.00153 | 5/5 | 3/5 |

Validity is exactly `1.0` on every seed, functional watertightness ranges
`0.99389-1.0`, and outside-footprint fraction is `0.0`. The opening regression
is a real tradeoff and remains a limitation.

The final aggregator decision is `hold`: the full-val and structural gates pass,
but offset-128 instance F1 improves on only one of five seeds and type accuracy
does not improve on any holdout seed. Do not promote the RGB dense-query room
identity readout as a general predicted-control SOTA advance.

## Engineering Repairs

- Added a typed adapter for independent dense and query checkpoints.
- Added a complete structural consumer rather than importing room-only raster
  metrics into graph fields.
- Added per-seed checkpoint and taxonomy SHA-256 records.
- Added a five-seed aggregator that preserves seed direction, CI, and best/worst
  seed instead of pooling cases.
- Added an explicit query-support probability and label-gating consumer.

## Support-Gate Audit

The previous learned support gate never changed deployed room support: labels
were always multiplied by the footprint, and `room_support_iou` stayed exactly
`0.765985` at every epoch. Its support loss therefore did not have a
metric-level output consumer. This is recorded as an engineering defect, not as
evidence against support gating itself.

The repaired diagnostic path now exposes query max/union support and can gate
room labels before metric computation. It must be screened on calibration and
offset-128 holdout before any learned support head is trained.

The fixed-threshold screen on seed 19 confirms that a naive support gate is not
promotable:

- val491 `blend50_union_or_conf0.80_0.03` changes count MAE by `-0.564`,
  but Room IoU by `-0.00149`, support IoU by `-0.01435`, and type accuracy by
  `+0.00058`;
- the offset-128 holdout gives the same qualitative tradeoff: instance F1 rises
  while Room IoU and type accuracy fall;
- no tested threshold improves all of Room IoU, support IoU, instance F1,
  count MAE, and type accuracy together.

This is a mechanism finding, not a model win. A follow-up head may only delete
low-confidence support if a hard preservation region keeps confident,
identity-correct room interiors unchanged.

Evidence:

- `outputs/reports/dense_query_support_seed19_cal64_v1.json`
- `outputs/reports/dense_query_support_seed19_holdout128_lowthr_v1.json`
- `outputs/reports/dense_query_support_seed19_cal64_confv1.json`
- `outputs/reports/dense_query_support_seed19_holdout128_confv1.json`
- `outputs/reports/dense_query_support_seed19_val491_confv1.json`

## Evidence

- `outputs/reports/dense_query_blend_seed{19..23}_{cal64,holdout128,val491}_v1.json`
- `outputs/reports/dense_query_blend_fiveseed_v1.json`
- `outputs/reports/dense_query_blend_seed{19..23}_structure_val491_v1.json`

