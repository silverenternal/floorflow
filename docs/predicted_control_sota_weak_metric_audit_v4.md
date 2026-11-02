# Predicted-Control SOTA Weak-Metric Audit v4

Date: 2026-11-02

## Decision

Predicted-control SOTA is still blocked. The authoritative full-val491 report
predates the current repair batch, so no repaired metric is claimed here.

| Metric | Current | Gate | Gap |
| --- | ---: | ---: | ---: |
| Room IoU | 0.327585 | 0.45 | 0.122415 |
| Instance F1 | 0.281759 | 0.50 | 0.218241 |
| Wall F1 | 0.487405 | 0.58 | 0.092595 |
| Opening F1 | 0.087022 | 0.35 | 0.262978 |
| Node F1 | 0.727429 | 0.82 | 0.092571 |
| Edge overlap | 0.361825 | 0.50 | 0.138175 |
| GED | 7.682281 | <= 6.0 | 1.682281 |

## Primary Bottleneck

Room support IoU is `0.774252`, while semantic Room IoU is `0.327585`.
Query-mask Dice is about `0.09` and instance F1 is `0.281759`. The model finds
most room foreground, but query masks, count-selected slots, owner assignment,
and semantic identity do not consistently refer to the same room instance.

Count errors propagate strongly:

| Count-error stratum | Cases | Room IoU | Instance F1 | Edge overlap |
| --- | ---: | ---: | ---: | ---: |
| exact | 90 | 0.364546 | 0.357672 | 0.445545 |
| 1-2 | 212 | 0.353409 | 0.300325 | 0.396023 |
| 3+ | 189 | 0.281019 | 0.224785 | 0.283599 |

Even count-exact cases remain below the instance and edge gates, so count
correction alone cannot close the problem.

## Confirmed Defects Repaired

1. Main class-plus-mask supervision, query presence, and program assignment used
   different matchings. They now share one `class + 2 * mask` Hungarian match.
2. Program-router foreground/background order did not match the main room-class
   head. The router now exposes explicitly ordered class assignment logits and a
   scheduled diagnostic contract.
3. Class-count prior and second-graph updates changed class logits after count
   selection had been computed. Count/presence selection is now resynchronized
   after each class-logit update.
4. The room graph scope enabled program assignment while freezing the program
   router. The logged program loss reached one element with a gradient norm near
   `5e-8`; the router is now in the room graph scope.
5. Dynamic balancing used non-gate `ce` as its reference and reduced the primary
   instance and partition scales to about `0.24` and `0.21` while auxiliary
   family scales reached `2.0`. Reference selection now follows the active
   training scope, and old checkpoint scales are reset when the reference moves.
6. `room_graph_adapter` freezes the room decoder. Its unchanged room support,
   precision, and recall across the six-epoch screen cannot diagnose query-mask
   or instance quality. A matched `room_adapter` control/candidate pair now
   exists for query-decoder and program-router repair.

The two-step CPU contract
`outputs/train/typed_channel_v6_room_identity_adapter_cpu_gradient_v3/report.json`
confirms the repaired paths are live: `instance_room` is the balance reference,
the sparse support-assignment objective is active, and the program term touches
1,100,691 parameter elements with gradient norm `1.56896` instead of the prior
one-element, approximately `5e-8` path. This is a contract check, not a metric
result.

## Follow-up Gate-Consumer Audit

The first matched room-adapter control completed six epochs and selected
epoch 1: Room IoU `0.311228`, Instance F1 `0.206919`, count MAE `4.90625`,
Node F1 `0.708879`, Edge overlap `0.333529`, and GED `7.59375`. Later epochs
degraded. Source:
`outputs/train/typed_channel_v6_room_identity_control_screen256_seed19_v1/report.json`.

The jointly configured arm was stopped after two epochs on user direction. Its
partial epoch 1 is not a matched result: Room IoU `0.310641`, Instance F1
`0.203329`, count MAE `4.625`, Node F1 `0.702718`, Edge overlap `0.322483`,
and GED `7.625`. No checkpoint is retained or promoted.

The screen exposed two missing diagnostics and two consumer defects:

1. Shared-match and count-gate diagnostics were computed but dropped by the
   training scalar whitelist. A one-case contract now reports four valid
   targets, four shared matches, gate availability `1.0`, mask-only matching
   disagreement `1.0`, selected-target recall, and selected-background rate.
2. Count-selected slots were multiplied by query presence again in room fusion,
   owner availability, and graph presence. The second presence threshold could
   reject a query already selected by the count gate. Selection is now the
   authoritative consumer gate.
3. The independent count head remained near its center-8 prior. Query presence
   now contributes a bounded Poisson-binomial count prior through a learned
   scale, coupling top-k selection to the matched query states.
4. Presence supervision had no direct positive-versus-background ranking term
   and count selection had no direct selected-slot alignment term. Both are now
   differentiable query-set objectives with independent balance entries.

The current two-case CPU contract reports `4.5` valid targets, `4.5` shared
matches, selected-target recall `0.7905`, selected-background rate `0.1884`,
and nonzero gradients for count, selection alignment, and presence ranking.
The same contract now also reports a direct normalized cardinality regression
from `sum(query presence)` to GT count and an active unmatched-query empty-mask
loss. Mask-only assignment disagreement is `0.9`, so choosing the class-aware
shared match is a substantive contract change rather than an implementation
detail. Source:
`outputs/train/typed_channel_v6_room_identity_query_gate_diagnostic_v12/report.json`.
These are contract values only. A fresh matched screen is required before any
quality claim.

## Query-Mask Diagnostic

The next audit isolates matched query-mask geometry from unmatched-query
leakage:

| Diagnostic | Value |
| --- | ---: |
| Matched mask Dice | 0.1395 |
| Matched mask IoU | 0.0789 |
| Matched predicted area | 0.0447 |
| Matched target area | 0.0626 |
| Unmatched predicted area | 0.00955 |

Matched masks have roughly the right area scale but low overlap. Unmatched masks
are not flooding the image, so adding empty-mask supervision alone cannot close
the query-mask gap.

The retained mask output head is also nearly zero (`std 0.00341`) compared with
the class head (`std 0.1134`). A deterministic scale-up to `std 0.02` was tested
and rejected: Dice fell `0.1395 -> 0.00406`, IoU fell `0.0789 -> 0.00185`, and
predicted area fell `0.0447 -> 0.00058`. The problem is not magnitude alone;
scaling an untrained spatial direction is destructive. The automatic warm-start
was removed.

A second mask-geometry candidate reused the existing prototype projection and
mask embedding to inject native-resolution room-control evidence. It was also
rejected on the same contract: matched Dice fell `0.13949 -> 0.12483`, IoU
`0.07888 -> 0.07013`, and predicted area `0.04473 -> 0.03725`. The raw control
does not supply a sufficiently reliable boundary signal by itself. The
implementation remains optional but is disabled in the matched candidate.

The matched-mask recall direction was then fixed with an independent room-mask
positive weight and Tversky term. On the same two-case contract, Dice improves
`0.13949 -> 0.14236`, IoU `0.07888 -> 0.08063`, and predicted area
`0.04473 -> 0.04647`. The mask-head gradient remains live (`0.56868`), while the
learned query-embedding gradient is still much smaller (`0.000657`). This is
evidence that the loss direction is now helping, not proof of a validation gain.

Sources:
`outputs/train/typed_channel_v6_room_identity_query_gate_diagnostic_v13/report.json`,
`outputs/train/typed_channel_v6_room_identity_query_gate_diagnostic_v14/report.json`,
and `outputs/train/typed_channel_v6_fullres_mask_cpu_contract_v3/report.json`.
The mask-direction contract is
`outputs/train/typed_channel_v6_mask_direction_contract_v1/report.json`.

## Query-Target Spatial Error

The next diagnostic separates localization from boundary quality. Matched mask
centroid error is `0.23222` in normalized image coordinates, while predicted and
target area differ by less than two percentage points. The queries can emit
roughly the right amount of mask area in the wrong location.

Dense room-control saliency is not a useful instance-anchor source: whole-room
foreground coverage makes it nearly uniform, and prototype-saliency anchors move
the references by only `0.00240`.

Matched target masks now supervise adaptive-reference centroids during training.
The two-step contract reports reference-anchor loss `0.06611`, reference-head
gradient `0.01060`, and reference-context gradient `0.06891`. Centroid error is
unchanged after two steps, so this establishes a live optimization path rather
than a result. Sources:
`outputs/train/typed_channel_v6_prototype_anchor_contract_v1/report.json` and
`outputs/train/typed_channel_v6_reference_anchor_contract_v4/report.json`.

Matched masks also receive the centroid-distance objective directly, so the
mask path is not forced to correct location only through an intermediate
reference head. The contract reports centroid loss `0.23222` and keeps the mask
head gradient live, but two optimizer steps do not materially change centroid
error. This is a bound optimization direction, not a result. Source:
`outputs/train/typed_channel_v6_mask_centroid_contract_v1/report.json`.

Adaptive-reference and query-position parameters now receive a separate
optimizer group at `10x` the base learning rate. On a matched 4-case contract
this does not materially change centroid error (`0.186762` versus `0.186762`)
or mask Dice (`0.124605` versus `0.124933`). Optimizer routing is therefore
valid but not sufficient by itself; the location path still needs structural
conditioning.

The zero-weight audit also found permanently unreachable MIta route-budget and
route-gate heads for the `envelope` stage. `envelope` has no predecessors, so
its route branch never executed even though the parameters existed. Those dead
modules were removed, and route modules are now instantiated only for stages
that have predecessor evidence.

## Remaining Absolute Blockers

- Query-mask quality, owner assignment, and type identity require a matched
  `room_adapter` screen and disjoint holdout. No full-val promotion exists yet.
- Wall precision remains the weak side of the wall layer. The existing portfolio
  candidate cache is not built from the complete stage wall plan, so training a
  selector over that cache cannot recover the missing geometry.
- Opening pixel F1 is about `0.205` while the same-case vector proxy is `0.0`.
  Pixel gains are not evidence for the CAD opening consumer.
- Node F1 is higher than Edge overlap. The fixed graph source is room identity;
  additional node-only heads are not the direct remedy.
- GED remains the downstream consequence of room count/type, edge
  insertion/deletion, and the fixed evaluator. It must be reported only in
  paired, evaluator-fixed comparisons.

## Next Promotion Gates

1. Run the matched room-identity control/joint screen on the same small subset
   and seed; retain the repair only if query-mask Dice, count, type, Instance F1,
   and Room IoU move in the same direction.
2. Confirm on a disjoint holdout before any full-val491 run.
3. Rebuild wall candidates from the complete final wall plan, then test
   precision-oriented selection without deleting recall.
4. Select opening checkpoints by vector F1, not pixel F1.
5. Re-run frozen-evaluator Node/Edge/GED only after room identity improves.
