# Structural Plausibility Metrics v1

## Decision

The predicted-control track is screened on structural plausibility before Room
IoU. IoU remains a diagnostic and is not a promotion gate for this phase.

## Reusable Evidence From Public Repositories

| Source | Reusable implementation | Decision |
|---|---|---|
| `third_party/fpvec-lab` | `fpeval.verify.verify`, wall/opening matching, room matching, topology-aware edit cost | Use directly through `floorflow.evaluation.fpeval_bridge` |
| Raster2Seq / RoomFormer evaluator | Room/corner/angle F1 definitions; 0.5 room IoU match, 10 px corner tolerance, 5 degree angle tolerance | Reuse the definitions; the class depends on dataset-specific `DataRW` and cannot be called directly on FloorFlow rows |
| GSDiff topology evaluator | Shared-edge room adjacency, room-type node substitution, NetworkX GED | Use as protocol reference; keep one fixed local evaluator for paired comparisons |
| CAGE planar-graph evaluator | Same room/corner/angle family as Raster2Seq, plus polygon refinement | Use definitions and overlap checks; do not run dataset loader |
| HouseGAN++ graph evaluation | Mask-to-room-graph extraction and GED | Diagnostic evidence only; its RPLAN assumptions do not match CubiCasa |
| CubiCasa FloorTrans metrics | Pixel accuracy and semantic mIoU | Insufficient for structural plausibility; do not promote |

Primary source links:

- Raster2Seq: https://github.com/Cornell-VAILab/Raster2Seq
- RoomFormer: https://github.com/ywyue/RoomFormer
- GSDiff: local `third_party/GSDiff`
- CAGE: local `third_party/CAGE`
- HouseGAN++: local `third_party/houseganpp`

## Plausibility Suite

### Intrinsic: no target geometry required

| Metric | Definition | Source |
|---|---|---|
| Parsable rate | Fraction of outputs that produce at least one wall and room face | `fpeval.verify` |
| Structural validity | Walls have legal ids/coordinates, openings attach to walls, room references resolve | `fpeval.verify` |
| Watertight rate | Room polygons form a connected partition without interior holes | `fpeval.verify` |
| Hole count | Number of raster holes inside the room union above frozen tolerance | `watertight_diagnostics` |
| Dangling endpoint rate | Fraction of wall endpoints not joined to another wall | `wall_junction_summary` |
| Room closure rate | Fraction of predicted room polygons that are valid closed polygons | `structural_legality_metrics` |
| Wall continuity rate | Fraction of junction clusters incident to at least two wall segments | `structural_legality_metrics` |
| Self-intersection rate | Fraction of predicted polygons that are invalid | `structural_legality_metrics` |
| Room overlap ratio | Pairwise predicted room intersection area divided by predicted room union | `structural_legality_metrics` |
| Out-of-boundary ratio | Predicted polygon area outside the frame divided by predicted room area | `structural_legality_metrics` |

### Comparative: target graph or target geometry required

| Metric | Definition | Source |
|---|---|---|
| Wall F1 | Angle, projection overlap, and symmetric segment-distance matching | `fpeval.metrics.wall_metrics` |
| Opening F1 | Door/window family and center matching with width ratio | `fpeval.metrics.opening_metrics` |
| Room-node F1 | Hungarian room matching at IoU 0.5 with type checking | `room_metric_suite` |
| Edge overlap | Shared-boundary adjacency precision/recall/F1 | `fpeval_metric_contracts.edge_precision_recall_f1` |
| Unit GED | Fixed NetworkX graph edit distance on room-type nodes and shared-edge adjacencies | local fixed evaluator |
| Edit cost | Topology-aware CAD correction cost | `fpeval.metrics.edit_cost` |
| Corner F1 | Predicted corner matched to target corner within 10 px | Raster2Seq definition |
| Angle F1 | Matched corner angle within 5 degrees | Raster2Seq definition |

## Reporting Rules

- Intrinsic plausibility and target-comparative accuracy are separate tables.
- Watertightness is always paired with hole count and dangling endpoints.
- Validity is reported as a conjunction plus per-check failure counts.
- Node F1, edge overlap, corner F1, angle F1, and GED use one frozen evaluator.
- Room IoU is reported as a diagnostic only and cannot block a plausibility screen.
- Dataset-specific evaluator classes are adapted through local row adapters; no
  external evaluator is mixed with a different graph construction.

## Candidate Gate

A mechanism advances to the next screen when:

1. parsable rate and structural validity do not regress;
2. watertightness improves without a material increase in holes or dangling
   endpoints;
3. room overlap and out-of-boundary ratios remain at zero or within the frozen
   tolerance;
4. wall continuity and edge overlap do not regress simultaneously;
5. the result is repeatable on the 256/64 mechanism screen.
