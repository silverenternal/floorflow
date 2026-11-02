# r156 implementation report (2026-10-25)

All seven r156 tasks are implemented. Full suite: **435 passed, 1 skipped**;
changed files pass `ruff check`; max Python file is 597 lines.

## Enhancements

- A0: `FootprintCrossChannelPrior` now has a boundary-mask projection and a
  boundary MultiheadAttention residual with zero-init `boundary_scale`, so old
  checkpoints load as identity and the branch is trainable.
- A1: `ReadoutWall` carries a continuity score; `reconcile_plan` uses
  `_close_gaps` (collinear/T-junction bridge) instead of unary-union
  re-noding, so clean walls are never split.
- A2: `WallIntervalHead` now has `interval_targets_from_plan` and
  `wall_interval_loss` (presence/start/length/type) with gradient tests.
- A3: `joint_distribution_cfm_loss` adds `furniture_vector_weight=0.05` and
  `furniture_family_weight=1.25`.

## Sweet-spot study (A4)

Fine-tuned from the converged checkpoint for 24 steps over train sizes
512/1024/2048/5126:

| Train size | room macro IoU | wall F1 | opening F1 | composite |
|---:|---:|---:|---:|---:|
| 512 | 0.1367 | 0.4378 | 0.1184 | 0.2310 |
| 1024 | 0.1369 | 0.4322 | 0.1314 | 0.2335 |
| 2048 | 0.1353 | 0.4336 | 0.1056 | 0.2248 |
| 5126 | 0.1352 | 0.4378 | 0.1275 | 0.2335 |

The curve is flat: knee at 512, last-doubling gain 0.0087 (< 0.01). At this
fine-tuning budget the model is **not data-limited**. The val-size study
picked **64** (bootstrap stability 0.992, composite bias 0.003 vs full val191).
Protocol caveat: this measures marginal data value from a converged
checkpoint, not from-scratch data scaling.

## Retraining (B0)

Trained at 512 train / 64 val for 128 steps with no gradient balancer,
`flow/pixel/element = 0.2/1.0/0.5`, head LR 2e-5.

- val64: room +0.0009, opening F1 +0.0093, type accuracy +0.0062, furniture
  -0.0022, composite +0.0031.
- test503: room +0.0005, room polygon +0.0004, instance geometric +0.0011,
  but opening F1 -0.0087, furniture -0.0033, balanced CAD 0.3293 vs 0.3303.

The val64 gain did **not** transfer to test503. The r156 candidate is rejected
and the r155 candidate remains frozen (`outputs/cvpr_package_v7_zero_human`).
This negative transfer is exactly why the final freeze uses val selection plus
a single test503 confirmation.

## Metric hierarchy (B1)

`metric_hierarchy_test503_v5.json` has no missing L0-L5 fields and complete
18-module ownership. `audit_geometry_hop` now reports unclassified-opening
count and fraction alongside the consistency checks.

## Evidence

- `outputs/reports/scaling_sweet_spot_v1.json`
- `outputs/reports/r156_retrain_val64_v1.json`
- `outputs/reports/r156_test503_v1.json`
- `outputs/reports/metric_hierarchy_test503_v5.json`
- `outputs/cvpr_package_v7_zero_human/submission_checklist.json`
