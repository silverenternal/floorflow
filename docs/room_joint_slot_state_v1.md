# Joint Room-Slot State v1

Date: 2026-11-01

## Motivation

The previous count routes improved count MAE by suppressing or deleting room
queries. Room IoU and instance F1 then regressed because the deleted queries
still carried valid room support. Independent count heads, post-hoc owner
reassignment, and count gates did not survive disjoint holdouts.

This branch changes the representation contract: class and presence are
conditioned on one shared slot state, and cardinality is derived from slot
presence rather than predicted by a separate count head.

## Implementation

`RoomSlotStateHead` consumes:

- decoder query features;
- full-resolution projected features;
- soft query-mask descriptors;
- slot overlap and class entropy;
- the parent query class logits.

Each slot state goes through slot self-attention. Shared residuals update its
class and presence logits. The count distribution is the Poisson-binomial
distribution implied by the slot presence logits:

```text
P(count = k) = coefficient of z^k in product_q((1-p_q) + p_q z)
```

The class and presence residuals are zero-initialized. At initialization, the
slot branch preserves the parent query class logits exactly. The count output
is derived from the existing class-existence probabilities, not from a newly
initialized independent count classifier.

The branch does not enable `query_count_gate` by default and does not apply a
query deletion mask to fused room logits. Count and owner geometry therefore
remain observable separately during screening.

## Integration

- `RasterInstanceControl` exposes `--joint-room-slot-state`.
- The legacy `--query-set-head` and joint slot state are mutually exclusive.
- `room_query_set_loss` supervises slot presence and the derived count
  distribution.
- Checkpoint loading restores the branch flag and rejects legacy architectures
  only by the existing strict state-shape contract.
- Class-only recovery can train the joint slot head without unfreezing proposal
  geometry.

## Smoke Evidence

Parent checkpoint:

```text
outputs/train/room_query_blend50_seed19_v1/best.pt
```

A two-train/two-validation `cuda:0` smoke completed training, validation,
blend selection, and checkpoint/report generation:

```text
outputs/train/room_joint_slot_state_smoke2_v1/report.json
```

The contract recorded `joint_room_slot_state: true`, active `query_set` loss,
and finite presence/count objectives. This smoke makes no quality or SOTA
claim.

## Next Screen

The matched mechanism screen is defined in
`configs/room_joint_slot_state_screen_v1.json`. Promotion requires the same
direction on train256/cal64 and a disjoint holdout64, including preservation of
room support. A count-only improvement with support loss is an automatic
reject.

## Screen Result

The first calibration result was positive for the full parent fine-tune:
Room IoU `+0.00239`, instance F1 `+0.01277`, count MAE `-0.03125`, and type
accuracy `+0.03624`. The disjoint holdout reversed the identity direction:
Room IoU `-0.00152`, instance F1 `-0.00210`, and type accuracy `-0.02641`,
while count MAE improved by `0.26563`.

A second arm froze every parent parameter and trained only the 18 tensors in
the joint slot head. Its holdout result remained mixed: Room IoU `+0.00554`
and type `+0.00781`, but instance F1 `-0.01282` and count MAE `+0.04688`.

Decision: reject the current slot-state head as a promotion candidate. Keep
the implementation as an isolated experiment, but do not enable it in the
canonical model. The next version must add a slot-conditioned mask/owner
residual and a frozen-parent class anchor; count weighting and query deletion
are not the next lever.

Evidence:

- `outputs/reports/room_joint_slot_state_screen_v1.json`
- `outputs/train/room_joint_slot_state_screen256_seed19_v2/report.json`
- `outputs/train/room_joint_slot_slotonly256_seed19_v1/report.json`
- `outputs/reports/room_joint_slot_parent_holdout64_v1.json`
- `outputs/reports/room_joint_slot_v2_holdout64_v1.json`
- `outputs/reports/room_joint_slot_only_holdout64_v1.json`
