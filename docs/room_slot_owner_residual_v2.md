# Room Slot-Owner Residual v2

Date: 2026-11-01

## Why v2 Exists

Joint room-slot state v1 changed class and presence but left query masks
unchanged. Its calibration gains reversed on disjoint holdout, and freezing the
parent still produced a Room IoU/type versus F1/count trade. The v1 branch is
registered as rejected.

Version 2 changes the owner geometry directly:

- the shared slot state produces a full-resolution query-mask residual;
- the residual is zero-initialized, so the parent mask can be recovered exactly;
- slot masks and a learned background form an exclusive owner distribution;
- `slot_owner` fusion composes owner probability with room-class probability;
- class residual magnitude is regularized by a frozen-parent anchor.

Count is still derived from slot presence. No count threshold or query deletion
mask is applied.

## New Outputs

- `mask_logits`: parent mask plus the slot-conditioned full-resolution residual.
- `slot_owner_probability`: mutually exclusive query-owner probability, with the
  background channel excluded.
- `slot_class_residual`: direct residual used by the class anchor.
- `slot_state_diagnostics.mask_residual_norm`: runtime geometry-residual scale.

## Loss Contract

`room_query_set_loss` gains `slot_class_anchor_weight`. It penalizes the squared
class residual, keeping class changes tied to the parent instead of allowing
small-data class drift. The mask residual remains supervised by the existing
Hungarian mask/class, fused-identity, and partition losses.

## Screen Contract

The next matched screen is
`configs/room_slot_owner_screen_v2.json`. It freezes the parent to isolate the
new slot head and evaluates:

1. frozen parent;
2. slot-owner residual without the class anchor;
3. slot-owner residual with the class anchor;
4. slot-owner residual without count supervision.

At least one of Room IoU or instance F1 must improve on both calibration and
disjoint holdout without the other regressing. A count-only or calibration-only
gain is rejected.

## Claim Boundary

This version passed implementation tests and a two-case GPU smoke, but the
matched screen failed:

| Split | Room IoU delta | F1 delta | Count MAE delta | Type delta |
|---|---:|---:|---:|---:|
| cal64 | +0.00348 | -0.00623 | -0.23438 | +0.00469 |
| holdout64 | -0.00669 | +0.00508 | -0.09375 | -0.02814 |

The identity direction reverses across splits. The branch is rejected and is
not enabled in the canonical model.

Evidence:

- `outputs/reports/room_slot_owner_screen_v2.json`
- `outputs/train/room_slot_owner_smoke2_v1/report.json`
- `outputs/train/room_slot_owner_screen256_seed19_v1/report.json`
- `outputs/reports/room_slot_owner_holdout64_v1.json`

## Dense Owner Assignment Follow-up

Adding a direct Hungarian-matched owner-pixel cross entropy did not repair the
mask identity. The best calibration checkpoint produced:

| Room IoU | Instance F1 | Count MAE | Type accuracy | Owner exact |
|---:|---:|---:|---:|---:|
| 0.25883 | 0.23024 | 3.46875 | 0.55469 | 0.66547 |

Against the parent, Room IoU rises `+0.00334` and count improves `-0.03125`,
but instance F1 falls `-0.01273` and type falls `-0.01076`. The run stops at
calibration; no holdout is opened for this arm.

Evidence:

- `outputs/reports/room_support_assignment_screen_v1.json`
- `outputs/train/room_slot_owner_assign256_seed19_v1/report.json`
- `outputs/train/room_slot_owner_assignment_smoke2_v1/report.json`
