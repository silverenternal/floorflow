# Room Instance Control v1

## Decision

The existing direct-GT `RasterInstanceControl` path is more promising than a
higher fixed-resolution run: its query masks improve instance F1, count, type
accuracy, and graph consistency when used as hard regions while dense logits
remain the class source. Full val also exposes a significant topology
regression, so the route is retained as evidence for a learned dense/query
arbitration task rather than promoted as the main endpoint. No joint-encoder or
global-blend variant is promoted.

## Pre-Execution Review

This run reuses `RasterInstanceControl`, `RoomMaskDecoder`,
`room_mask_loss`, `room_identity_loss`, `mask_region_class_logits`, the existing
trainer, and the shared explicit-plan evaluator.

The previous teacher-distillation attempt is not evidence against this route:
PC-04 copied class-agnostic Raster2Seq proposals, while this run directly
supervises query masks and classes from Ground Truth. The current experiment uses
the canonical 384x384 cache, the selected native-raster room checkpoint, and no
teacher at training or inference.

## Frozen-Encoder Screen

The existing selection score was corrected before interpreting the result. The
original score could select a blend with near-zero instance F1 because a type
bonus outweighed the identity loss. The new score is:

`Room IoU + 0.75 * Instance F1 + 0.1 * Type accuracy - 0.02 * count MAE`

At the best 8-epoch checkpoint, the internal train256/cal64 screen reports:

| Arm | Room IoU | Instance F1 | Count MAE | Type accuracy |
|---|---:|---:|---:|---:|
| Frozen dense branch | 0.24685 | 0.21706 | 3.93750 | 0.53058 |
| Query blend 0.75 | 0.25608 | 0.24387 | 3.42188 | 0.55469 |

The same configuration was run to 10 epochs. Its best checkpoint is epoch 8;
epoch 9 regresses, so the documented plateau occurs after ten epochs rather than
requiring the full 24-epoch promotion budget.

## End-to-End Calibration

The dense checkpoint and instance query decoder are evaluated through the same
predicted structural bridge and explicit-room legalizer.

| Route | Room IoU | Instance F1 | Count MAE | Type accuracy | Node F1 | Edge F1 | GED | Plan plausible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense endpoint | 0.33170 | 0.26925 | 2.5625 | 0.68099 | 0.74349 | 0.49381 | 9.8125 | 0.96875 |
| Probability blend 0.25 | 0.33015 | 0.26747 | 2.5938 | 0.69818 | 0.75726 | 0.51402 | 9.2344 | 0.96875 |
| Probability blend 0.50 | 0.32631 | 0.26955 | 2.4375 | 0.68125 | 0.75096 | 0.50208 | 8.6094 | 0.95313 |
| Class-region, threshold 0.5 | 0.33052 | 0.27010 | 2.4063 | 0.72578 | 0.73980 | 0.49203 | 9.3750 | 0.96875 |
| Class-region, threshold 0.6 | **0.33258** | **0.27354** | **2.3438** | **0.72578** | **0.74217** | **0.49108** | **9.2656** | **0.95313** |
| Class-region, threshold 0.7 | 0.33211 | 0.26544 | 2.4531 | 0.69063 | 0.74163 | 0.49287 | 9.3125 | 0.93750 |
| Class-region, threshold 0.8 | 0.33100 | 0.27128 | 2.4219 | 0.69063 | 0.74057 | 0.49165 | 9.5938 | 0.96875 |

Threshold 0.6 is the only tested arm that improves both Room IoU and instance
F1 while lowering count error on calibration64.

## Offset-128 Check

| Route | Room IoU | Instance F1 | Count MAE | Type accuracy | Node F1 | Edge F1 |
|---|---:|---:|---:|---:|---:|---:|
| Dense endpoint | 0.39676 | 0.31516 | 3.2500 | 0.81328 | 0.78496 | 0.57236 |
| Class-region, threshold 0.5 | 0.39430 | 0.32319 | 3.0313 | 0.80547 | 0.79063 | 0.57750 |
| Class-region, threshold 0.6 | 0.39544 | 0.32339 | 2.9531 | 0.80547 | 0.79063 | 0.57800 |
| Class-region, threshold 0.7 | 0.39571 | 0.32133 | 3.0625 | 0.80807 | 0.79063 | 0.57800 |
| Class-region, threshold 0.8 | 0.39598 | 0.31828 | 3.0938 | 0.81328 | 0.79063 | 0.57584 |

The offset arm confirms the instance/count/graph direction but shows a room-IoU
regression of roughly 0.001.

## Full Validation

The prespecified threshold 0.6 arm was evaluated on all 491 validation cases:

| Metric | Dense endpoint | Query class-region | Paired result |
|---|---:|---:|---|
| Room IoU | 0.33386 | 0.33293 | delta -0.00093; 95% CI [-0.00349, 0.00143], p=0.710 |
| Instance F1 | 0.28883 | 0.29451 | delta +0.00568; 95% CI [-0.00061, 0.01207], p=0.038 |
| Count MAE | 3.59267 | 3.39104 | delta -0.20163; 95% CI [-0.31573, -0.08758], p=0.0018 |
| Type accuracy | 0.68362 | 0.68574 | delta +0.00213; CI crosses zero |
| Node F1 | 0.71271 | 0.71629 | delta +0.00358; p=0.029 |
| Edge F1 | 0.46500 | 0.47075 | delta +0.00575; CI above zero |
| GED | 11.24236 | 10.72709 | delta -0.51527; p=2.4e-13 |
| Wall continuity | 0.98856 | 0.98645 | delta -0.00211; p=0.00038 |
| Plan plausible | 0.95519 | 0.92872 | delta -0.02648; p=0.0029 |

The query route improves identity count, graph metrics, and GED, while Room IoU
is statistically indistinguishable from dense. However, it significantly
degrades plan plausibility and wall continuity, increasing wall-continuity
failures from 22 to 35 cases and non-watertight plans from 222 to 233 cases.
It cannot be promoted as the main fidelity endpoint.

Keeping the original dense logits as the structural-bridge input did not remove
the failure on calibration64: Room IoU remained 0.33263, instance F1 0.27248,
and the graph metrics were unchanged. The harmful coupling is therefore in the
routed room partition, not in feeding that partition to the wall model.

## Joint-Encoder Budget Audit

Jointly training the encoder and query decoder at the decoder learning rate
collapsed Room IoU to 0.050 in the first epoch. Adding a lower encoder learning
rate of `2e-5` avoided catastrophic collapse but still stayed below the frozen
encoder arm for all eight epochs. The joint path is therefore rejected on the
fixed screen budget; this is an optimization/regime mismatch, not evidence that
query masks are useless.

## Evidence

- `outputs/train/raster_instance_control_384_screen256_v1/report.json`
- `outputs/train/raster_instance_control_384_promotion256_10ep_v1/report.json`
- `outputs/train/raster_instance_control_384_joint_low_lr_screen256_v1/report.json`
- `outputs/reports/predicted_control_affinity_10ep_screen64_blend05_v1.json`
- `outputs/reports/predicted_control_affinity_10ep_class_region64_v1.json`
- `outputs/reports/predicted_control_affinity_10ep_class_region64_t060_s8_v1.json`
- `outputs/reports/predicted_control_affinity_10ep_class_region_holdout64_offset128_t060_s8_v1.json`
- `outputs/reports/predicted_control_affinity_10ep_class_region_val491_v1.json`
- `outputs/reports/predicted_control_affinity_10ep_class_region_val491_stats_v1.json`
- `outputs/reports/predicted_control_affinity_10ep_class_region64_t060_s8_dense_structure_v1.json`
