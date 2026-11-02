# Raster2Seq + FloorFlow Structural Audit v2

## Scope

This is a clearly labeled hybrid audit on Raster2Seq's official validation
split. It is not a standalone FloorFlow result and does not enter the
standalone predicted-control leaderboard.

The hybrid keeps Raster2Seq proposal rooms and openings, then applies the
FloorFlow structural contract:

- room proposals become the room layer;
- one-pixel seams become interior wall centerlines;
- unsupported openings are removed;
- known exterior structure is composed at width 7;
- explicit room polygons and gap-reconciled wall graphs are audited.

The original Raster2Seq Room/Corner/Opening values come directly from its
official per-case result JSON files. Project rasterization values are labeled
`adapter_*` and are not substituted for the official scorer.

## Results

Raster2Seq official val503, `503` cases:

| Metric | Value |
|---|---:|
| Official Room IoU | 0.63228 |
| Official Room precision / recall | 0.88801 / 0.88927 |
| Official semantic Room F1 | 0.64711 |
| Official Corner F1 | 0.59368 |
| Official Angle F1 | 0.37020 |
| Official opening F1 | 0.34026 |

Project adapter and FloorFlow completion:

| Metric | Value |
|---|---:|
| Adapter instance F1 | 0.83585 |
| Adapter type accuracy | 0.58870 |
| Adapter count absolute error | 1.2485 |
| Predicted / target room count | 8.795 / 8.871 |
| Room coverage | 0.99894 |
| Raster plausible | 0.99801 |
| Structural validity | 1.00000 |
| Componentwise wall closure | 0.99718 |
| Functional watertightness | 0.89264 |
| Wall continuity | 0.97916 |
| Plan plausibility | 0.80716 |

## Interpretation

Raster2Seq supplies much stronger room identity than the current standalone
FloorFlow native branch:

- instance F1 `0.836` versus FloorFlow `0.231`;
- room-count error `1.249` versus FloorFlow `3.218`;
- the official Room IoU is `0.632`.

FloorFlow structural completion preserves parseability and five-check validity
for every case and removes unsupported openings, but only `89.3%` of cases reach
functional watertightness. Raster2Seq room unions often contain several
disconnected spatial components, so seam/exterior composition is not yet a
complete legalizer for this proposal family.

The split sizes and control contracts differ:

- this audit: Raster2Seq val503;
- standalone FloorFlow baseline: FloorFlow val491.

No direct ranking between those two rows is allowed.

## Decision

Keep the hybrid as an external-threat and diagnostic path, not a standalone
SOTA claim.

The result supports two concrete next routes:

1. Improve FloorFlow legalization for multi-component Raster2Seq room unions,
   targeting functional watertightness and plan plausibility above `0.95`
   without changing official Room metrics.
2. Use the proposal identity signal as a teacher for a future FloorFlow
   instance decoder, while retaining a separately matched standalone row.

Machine-readable evidence:

- `outputs/reports/raster2seq_floorflow_audit_val503_v2.json`
