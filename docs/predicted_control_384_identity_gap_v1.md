# Predicted-Control 384px Identity Gap v1

## Decision

The 384px checkpoint remains the strongest measured room-identity frontier, but
the next move is not another 384px epoch increase. The full validation split
shows that most recoverable error is in over-segmentation, count correction, and
semantic assignment. A budget-gated 512px probe is justified for fragmented
instances, while a separate type-conditioned instance-affinity route is needed
for the larger semantic-mismatch component.

## Provenance Audit

The earlier `predicted_control_fidelity_384_*_v1.json` reports omitted
`room_size`, model-root, and several decode parameters. The canonical replay
below records the full configuration and the correct 384px target cache:

| Run | Room IoU | Instance F1 | Count MAE |
|---|---:|---:|---:|
| Historical `v1` report | 0.33320 | 0.28763 | 3.57026 |
| Canonical `decomp_v2`, explicit 384 cache and room size 224 | 0.33386 | 0.28883 | 3.59267 |

The canonical row is now the evidence source. The small historical difference is
consistent with the omitted inference contract rather than a change in the
384px training conclusion. All promoted numbers must come from reports that
record every inference parameter and seed.

## Full-Case Decomposition

The canonical replay uses all 491 validation cases and no target-derived input.

- 310 cases over-predict room count and 126 under-predict; only 55 have exact
  count.
- There are 1,473 excess predicted instances and 291 missing instances, for a
  net excess of 1,182.
- The over-major stratum has 192 cases, Room IoU 0.28613, instance F1 0.25765,
  and contributes 20.65 Room-IoU points and 16.25 instance-F1 points of gate
  deficit.
- The under-major stratum has 41 cases, Room IoU 0.29525, instance F1 0.17449,
  and contributes 5.31 Room-IoU points and 6.67 instance-F1 points.
- The 126 cases with at least 11 target rooms average Room IoU 0.25698 and
  instance F1 0.21350.
- The 55 count-exact cases still average only Room IoU 0.39761 and instance F1
  0.36469, so count correction alone cannot close the gate.

Pixel allocation over the room union is:

| Allocation | Fraction |
|---|---:|
| Correct room identity | 0.47802 |
| Semantic mismatch | 0.31209 |
| Spurious room prediction | 0.15990 |
| Missing room prediction | 0.04999 |

The highest-deficit classes include `room:Undefined`, `room:Elevated`,
`room:Sauna`, `room:Alcove`, `room:Room`, `room:Hall`, `room:Closet`,
`room:Bath`, `room:UserDefined`, and `room:Office`. Common confusions are
Bedroom/LivingRoom/Kitchen/Undefined, and Bedroom/LivingRoom into non-room.

## Pre-Execution Budget Review

| Candidate | Evidence | Budget | Gate | Decision |
|---|---|---|---|---|
| Extend 384px | 40-epoch continuation peaked at Room IoU 0.31839, below 32 epochs | 8 more epochs | Any positive val491 direction | Rejected |
| Generic fragment merge | Minimum area 64 changes no calibration64 room metric | No training | Count/identity gain | Rejected |
| Matched 512px probe | 384px improved every target metric over 256px; high tiny-component strata still fail | Cache train256/val256, 8-epoch train256/cal64 screen | Gain Room IoU and instance F1 without type/count regression | Run next |
| Type-conditioned affinity | 31.2% semantic mismatch, 16.0% spurious and count-exact cases remain weak | Reuse 384 features; 8-epoch screen, then 24 or plateau | Lower count MAE, raise Room IoU/F1, improve type consistency | Plan after 512 verdict |

The matched 512px arm must use the same warm start, seed, optimizer,
augmentation, class balance, batch policy, NFE, decode, and case order as its
384px control. A flat or negative 8-epoch curve keeps 384px and transfers the
budget to the identity-specific mechanism. A positive curve receives a 24-epoch
promotion run, not an immediate full-train jump.

## Next Contract

1. Freeze the canonical full-val report and its complete config.
2. Run the matched 384/512 screen on train256/cal64.
3. If 512 is positive, promote to train5126 and integrate through the existing
   structural bridge and explicit-room legalizer.
4. In parallel, design a type-conditioned instance-affinity head over existing
   encoder features. It must be connected to the explicit-plan output and pass
   end-to-end gradient and reasonableness audits.
5. Keep test503 closed.
