# Room Mechanism Screen v101

Date: 2026-11-01

## Decision

Reject the following query-head fixes as promotion candidates:

- longer training at train1024/val256;
- mask-pooled class fusion;
- local full-resolution class sampling;
- parameter-free owner semantic reassignment and its fine-tune screen;
- joint full-resolution owner and boundary refinement.
- room-count/complexity curriculum sampling.
- supervised same-class query separation.
- explicit query-set count gating.
- mask-pooled residual scale 1.0.

They converge on the same failure pattern: count MAE improves while Room IoU
and/or instance F1 regresses. This is not evidence for a stronger room-identity
model.

## Parent

The matched train256/cal64 parent at checkpoint-selected blend `0.75` has:

| Room IoU | Instance F1 | Type accuracy | Count MAE |
|---:|---:|---:|---:|
| 0.25567 | 0.24932 | 0.55443 | 3.453 |

## Screens

| Arm | Room IoU | Instance F1 | Type accuracy | Count MAE | Decision |
|---|---:|---:|---:|---:|---|
| Parent | 0.25567 | 0.24932 | 0.55443 | 3.453 | Reference |
| Train1024 holdout128 | -0.00748 Δ | +0.00015 Δ | -0.00469 Δ | -0.453 Δ | Reject |
| Mask-pooled class fusion v2 | 0.24869 | 0.24986 | 0.55295 | 3.203 | Reject |
| Full-resolution sampler | 0.24869 | 0.24986 | 0.55295 | 3.203 | Reject |
| Same-class separation | 0.25311 | 0.23671 | 0.55998 | 3.125 | Reject |

The train1024 comparison uses 128 untouched holdout cases. Its Room IoU CI
crosses zero but is directionally negative; instance F1 is flat; only count MAE
has a CI below zero.

## Interpretation

The query path currently buys count calibration by deleting or suppressing
fragments. Those fragments still carry useful semantic room area, so the result
is a trade rather than an identity repair. More budget and generic mask/class
regularization do not change that optimization behavior.

The next implementation must preserve the dense semantic class evidence and
assign it to explicit room owners at full resolution. A count head or
threshold-only readout is not sufficient.

The full-resolution sampler's epoch-7 checkpoint was also evaluated on the
disjoint offset-64 holdout: Room IoU `-0.009996`, instance F1 `-0.026338`, and
count MAE `-0.5625` relative to the dense model. The failure is therefore not
calibration-only.

Using the gate-aligned epoch-7 blend `1.0` increased the holdout regression to
Room IoU `-0.025118` and instance F1 `-0.026225`, despite a count MAE improvement
of `-0.59375`. The class-evidence residual family is closed for this readout.

Owner semantic reassignment was tested zero-shot and with parent fine-tuning.
On cal64 it changed Room IoU by `-0.00290`, instance F1 by `-0.00076`, and count
MAE by `+0.1875`. On holdout64, Room IoU changed by `+0.00257` but its CI crossed
zero, while instance F1 changed by `-0.00279` and count MAE by `+0.1094`.
Fine-tuning produced no non-zero query blend that beat dense on Room IoU/F1 and
met the predeclared negative-three-epoch stop rule.

The joint owner-boundary refiner was the first arm to improve all four
calibration metrics at epoch 2 / blend 0.75:

| Split | Room IoU Δ | Instance F1 Δ | Type Δ | Count MAE Δ |
|---|---:|---:|---:|---:|
| cal64 | +0.00604 | +0.01123 | +0.03831 | -0.71875 |
| holdout64 | -0.00142 | -0.01265 | +0.00296 | -0.76563 |

The disjoint holdout rejects promotion. This is retained as a concrete
calibration-generalization warning: a multi-metric calibration win is not
evidence unless the same direction survives the holdout.

The room-count/complexity curriculum was matched on architecture and update
budget but sampled 256 cases per epoch from a 1024-case complexity pool. Its
cal64 metrics stayed below the parent, and holdout64 produced Room IoU
`+0.00299` with a CI crossing zero, instance F1 `-0.00292`, and count MAE
`-0.65625`. Extra complex sampling alone does not repair instance identity.

The explicit query-set head was trained with a 100-step gate warmup so its
presence/count heads could learn before selecting queries. Forced activation of
the saved count gate produced:

| cal64 | Grouped change |
|---|---:|
| Room IoU | -0.08697 |
| Instance F1 | -0.03677 |
| Type accuracy | -0.13084 |
| Count MAE | -0.46875 |

Count improves by removing queries that still carry room structure. This branch
is rejected. Effective gate state is now checkpointed so a warmup checkpoint is
not silently evaluated with the gate enabled.

The initial query-set presence target incorrectly treated the first N target
slots as positive queries. Presence is now supervised by each query's best
target-mask IoU, removing target-order dependence. Repeating the screen after
this fix produced the same forced-gate collapse: Room IoU `-0.08697`, F1
`-0.03677`, type `-0.13084`, count MAE `-0.46875`. The branch is closed.

## Implementation Defects Fixed

The first mask-pooled screen did not learn. Both the residual output and the
multiplicative residual scale were initialized to zero, so the module was
identity but had zero gradient. The output remains zero-initialized. The
multiplicative scale is retained at `0.1`; raising it to `1.0` increased the
calibration noise (`Room IoU 0.25424`, instance F1 `0.24302`, count MAE
`3.53125`) without repairing identity, so the larger scale is rejected.

The checkpoint loader also omitted new head flags and read the wrong config keys
for adaptive refinement and dual controls. Loader parity is now covered by a
round-trip test.

The legacy checkpoint selector over-weighted count MAE. A new
`dense_relative_identity` mode maximizes the weaker of Room IoU and instance F1
gains, then uses count and type only as secondary terms. The legacy mode remains
available for reproducing historical runs.

Evidence:

- `outputs/reports/room_mechanism_screen_v101.json`
- `outputs/train/room_query_blend50_train1024_seed19_v1/report.json`
- `outputs/reports/room_query_train1024_seed19_holdout128_v1.json`
- `outputs/train/room_query_maskpool_screen256_seed19_v1/report.json`
- `outputs/train/room_query_maskpool_screen256_seed19_v2/report.json`
- `outputs/train/room_query_fullsamples_screen256_seed19_v2/report.json`
- `outputs/reports/room_query_fullsamples_last_holdout64_v2.json`
- `outputs/reports/room_query_fullsamples_last_blend1_holdout64_v2.json`
- `outputs/reports/room_query_owner_reassign_cal64_v1.json`
- `outputs/reports/room_query_owner_reassign_holdout64_v1.json`
- `outputs/reports/room_query_owner_reassign_earlystop_cal64_v1.json`
- `outputs/reports/room_query_owner_reassign_earlystop_holdout64_v1.json`
- `outputs/train/room_query_ownerboundary_screen256_seed19_v1/report.json`
- `outputs/reports/room_query_ownerboundary_screen256_seed19_holdout64_v1.json`
- `outputs/train/room_query_ownerboundary_curriculum256_seed19_v1/report.json`
- `outputs/reports/room_query_ownerboundary_curriculum_seed19_holdout64_v1.json`
- `outputs/train/room_query_set_screen256_seed19_v2/best.pt`
- `outputs/reports/room_query_set_gate_forced_cal64_v2.json`
- `outputs/train/room_query_maskpool_screen256_seed19_v3/partial_report.json`

The v1 mask-pooled report is retained only as evidence of the zero-gradient
dead initialization and is invalid as a mechanism comparison.
- `outputs/train/room_query_sepsame_screen256_seed19_v1/report.json`
