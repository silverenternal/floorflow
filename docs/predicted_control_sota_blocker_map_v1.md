# Predicted-Control SOTA Blocker Map v1

Date: 2026-11-02

This is an evidence-to-module audit, not a SOTA claim. Source:
`outputs/reports/predicted_sota_blocker_map_v1.json`.

## Absolute Gates

| Metric | Current | Gate | Gap | Primary owner |
| --- | ---: | ---: | ---: | --- |
| Room IoU | 0.327585 | 0.45 | 0.122415 | room queries, owner readout |
| Instance F1 | 0.281759 | 0.50 | 0.218241 | room query assignment/count |
| Wall F1 | 0.487405 | 0.58 | 0.092595 | wall stage and candidate precision |
| Opening F1 | 0.087022 | 0.35 | 0.262978 | opening host, interval, vector consumer |
| Node F1 | 0.727429 | 0.82 | 0.092571 | room-type identity and presence |
| Edge overlap | 0.361825 | 0.50 | 0.138175 | typed adjacency and room geometry |
| GED | 7.682281 | <= 6.0 | 1.682281 | Node/Edge errors, especially room count/type |

All seven remain blocking.

## Decomposition

Room support IoU is `0.774252`, while semantic Room IoU is `0.327585`. The
model finds most interior support; the primary loss is instance/type identity,
count, and owner assignment, not raster foreground recall.

Wall recall `0.675861` is much higher than precision `0.409074`. This is a
false-positive/selection problem. Adding recall-oriented geometry will not
close the gate by itself.

Opening is the largest absolute gap. The full-stage number predates the newer
center-width vector arm, but the current family diagnostic also shows low
opening precision. Host geometry and the final interval consumer must improve
together.

The new raster-to-vector proxy makes the mismatch explicit: on the same CPU
case, pixel Opening F1 is `0.2049` while vector Opening F1 is `0.0`. Pixel F1
can no longer silently select a checkpoint for the CAD consumer.

Node F1 is much higher than Edge overlap. The canonical graph source is room
polygons, not wall segments. Additional node heads are not the direct lever.

## Latest Graph-Only Screen

The matched `room_graph_adapter` control completed six epochs before the
user-directed stop. Its selected metrics were:

- Room IoU `0.309552`, precision `0.771119`, recall `1.0`
- Instance F1 `0.210030`, Type accuracy `0.658333`, Count MAE `4.484375`
- Node F1 `0.721957`, Edge overlap `0.341413`, GED `7.5625`

Room precision and recall were exactly invariant across all six epochs. Node
and Edge moved by only about `0.003` and `0.002`; Type accuracy regressed.
The graph residual path was therefore not changing final room support enough.

The identified initialization dead zone has been repaired: graph-to-mask and
graph-to-class residual heads are warm-started after loading a count-aware
second-graph checkpoint. Runtime diagnostics now record first/second class and
mask residual norms; the CPU contract reports class residuals around `0.57`
and mask residuals around `60`. A candidate screen is required before judging
that repair.

## Next Code Ownership

- Room/count/type: query-set class cardinality, selection, and owner readout.
- Graph: graph residual output magnitude and canonical graph consumer.
- Wall: precision selection after room-boundary handoff.
- Opening: learned interval and host connectivity through the vector consumer.

