# Room Learning-Head Optimization v2

The v2 predicted readout remains the confirmed baseline. Its next query-set
candidate is defined in `docs/room_query_head_mita_v3.md`.

## Scope

This change strengthens the residential room-query training path without
changing the chip/FloorSet code or running a GPU experiment. All new model
branches are opt-in and identity-initialized where they modify an existing
prediction.

## Repaired Data and Supervision Contract

- The canonical evaluator now reads the query blend stored by the selected
  checkpoint. An explicit CLI override is recorded with its source.
- The query trainer records an active-loss contract in every checkpoint/report.
- Frozen-encoder training no longer labels a fused-map loss as a dense loss.
  `fused_identity` supervises the fused output; `dense` is present only when the
  dense encoder is trainable.
- The query trainer can use a deterministic shuffled small-data subset instead
  of always training on the file-order prefix.
- Gradient-balancer EMA and scale state are restored on warm start.

## Added or Activated Learning Heads

| Capability | Implementation | Consumer |
| --- | --- | --- |
| Dense identity residual | existing `PredictedRoomIdentityHead` | dense logits and blend |
| Mask-region class evidence | `MaskPooledClassFusion` | query class logits, fused room map |
| Query existence | `RoomQuerySetHead` | balanced presence BCE and optional existence gate |
| Instance count | `RoomQuerySetHead.room_count_logits` | count CE and checkpoint diagnostics |
| Per-class count | `RoomQuerySetHead.room_class_count_logits` | per-class cardinality CE |
| Room program | existing `RoomProgramSetHead` | set, presence, cardinality, co-occurrence losses |
| Same-class overlap | existing `RoomQuerySeparation` objectives | matched query masks |
| Query partition ownership | existing `query_partition_loss` | partition-consistent decoder |
| Count-aware query support | `query_count_candidate_mask` | optional `query_count_gate` |
| Full-resolution identity evidence | `MaskPooledClassFusion` and adaptive references | class evidence and query updates |
| Confidence-weighted partition | existing `room_partition_confidence` | occluded-room partition loss |

The adaptive reference sampler now receives the full-resolution encoder
feature map. The query decoder's attention memory remains half-resolution for
compute control, while identity evidence uses the full map through bounded
adaptive pooling or learned sampling.

## Compatibility

Old checkpoints load with all new branches disabled. New branches use
zero-initialized final projections, so enabling them before training preserves
the parent prediction exactly. New configuration keys are absent-safe.

## Reused Local References

- Mask2Former commit `9b0651c6c1d5b3af2e6da0589b719c514ec0d69a` for the
  existing masked query decoder and point-sampled mask supervision design.
- Deformable-DETR commit `11169a60c33333af00a4849f1808023eba96a931` for the
  existing sample-dependent adaptive query reference mechanism.
- LibMTL commit `4336804847eaa5e0b924b743d76beec7ac3fdc97` for the existing
  gradient-norm balancing objective.
- Existing `RoomProgramSetHead`, `room_query_separation`, and
  `query_partition_loss` were wired into the residential trainer rather than
  duplicated.

## Reproducible Entry Point

The implementation-ready budget and ablation contract is stored in
`configs/room_query_head_full_v2.json`. No associated GPU run has been started.

Relevant tests:

```bash
uv run pytest \
  tests/test_mask_pooled_class_fusion.py \
  tests/test_room_query_set_head.py \
  tests/test_room_mask_decoder.py \
  tests/test_room_segmenter.py \
  tests/test_raster_instance_blend_contract.py \
  tests/test_raster_instance_initialization.py -q
```

## Claim Boundary

These changes remove implementation and supervision bottlenecks. They do not
establish a new metric result until the screen, disjoint holdout, promotion, and
five-seed confirmation gates run.

## CPU Verification

- Targeted room-head suite: `39 passed`.
- Full non-chip test set: `845 passed`.
- Changed files pass Ruff.
- All non-chip Python files remain below 600 lines.
- The existing seed19 query checkpoint loads and runs on CPU with every new
  branch disabled.
