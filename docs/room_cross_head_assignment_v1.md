# Cross-Head Room Assignment v1

Date: 2026-11-01

## Finding

The unified floorplan generator used two independent exact assignments for the
same room-query slots:

- polygon supervision matched class plus polygon cost;
- mask supervision re-matched class plus mask cost.

A query could therefore be trained as one target room through its polygon head
and a different target room through its instance-mask head. This violates the
shared query-identity contract and can inject contradictory gradients into the
same slot.

## Repair

- `match_room_queries` can now include predicted and target masks in one joint
  assignment cost.
- `unified_generator_loss` computes one assignment from class, polygon, and
  mask evidence, then passes the same matches to polygon and mask supervision.
- `room_mask_loss` accepts precomputed shared matches and no longer re-solves a
  second assignment when they are provided.
- The mask cost lives in `floorflow.objectives.room_assignment_cost`; objective
  code no longer imports a loss implementation from a head module.
- CPU-collated target counts come from valid target slots, so a truncated
  target tensor cannot produce an oversized assignment count.
- `room_mask_metrics` can use the same class/mask assignment weights as
  training, removing a metric/training matching mismatch.

The implementation is opt-in through `--enable-joint-room-assignment`. The
canonical default remains independent because the matched quality screen
rejects joint assignment. `--joint-mask-match-weight 0.0` tests shared
polygon-only matching when joint assignment is enabled.

## Zero-Training Readout Audit

On the fixed seed19 query checkpoint and calibration64:

| Readout | Room IoU | Instance F1 | Count MAE | Type accuracy |
|---|---:|---:|---:|---:|
| dense | .24685 | .21706 | 3.9375 | .53058 |
| blend50 | .25873 | .25185 | 3.7344 | .56504 |
| blend75 | **.26195** | .24889 | 3.4531 | .59852 |
| query | .25620 | .24281 | 3.4219 | .62613 |
| query mixture | .25620 | .24281 | 3.4219 | .62613 |
| hard query | .25347 | .24516 | 3.8281 | .57014 |
| winner query | .25495 | .24661 | 3.3906 | .61159 |
| legacy | .00048 | .11614 | 7.8281 | .00000 |

The audit rejects readout-only rescue. Existing query evidence, not the current
fusion expression, remains the bottleneck.

Evidence:

- `outputs/reports/room_query_readout_audit_cal64_v2.json`
- `outputs/reports/rplan_room_assignment_screen_fast64_v1.json`

## Contract Verification

An E1 `cuda:0` contract-smoke run (resolved to 2 train / 1 validation by the
contract harness) completed training, validation, gradient diagnostics, token
consumer, checkpoint, and report generation:

- `outputs/train/rplan_joint_identity_contract_v1/report.json`

Focused tests cover shared-match reuse, joint mask-cost assignment, truncated
count safety, and CLI ablation wiring.

## Next Quality Gate

A fixed-parent train64/validation32 screen was run:

1. independent assignment;
2. joint class+polygon+mask assignment.

The best joint validation selection score was `0.46046`, below independent
`0.46456`; both remained below the pretrained parent. Joint assignment is
therefore retained as an explicit structural ablation, not promoted. The next
room mechanism must improve class/mask query evidence rather than force one
assignment across representations whose geometry quality differs.

