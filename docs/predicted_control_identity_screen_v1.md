# Predicted-Control Identity Screen v1

## Frozen Baseline

The reference is DMD-v11 plus the independently trained room segmenter, decoded
through the existing distribution overlay. On CubiCasa val491:

- Room macro IoU: `0.170526`;
- instance geometric IoU: `0.511269`;
- instance F1: `0.161880`;
- instance type accuracy: `0.452478`;
- room support IoU: `0.798084`;
- validity: `1.000000`;
- watertightness: `0.906314`.

This is the target-free predicted-control baseline. It must not be confused
with the target-derived `0.9718` capacity bound.

## Screen Results

| Arm | Room IoU | Instance IoU | Instance F1 | Type accuracy | Decision |
|---|---:|---:|---:|---:|---|
| v11 + segmenter | 0.17053 | 0.51127 | 0.16188 | 0.45248 | reference |
| query-mask identity head, best deployable blend | 0.17104 | 0.51833 | 0.16457 | 0.45713 | insufficient gain |
| dense residual identity head, best blend | 0.17069 | 0.51064 | 0.16128 | 0.45187 | insufficient gain |
| region-class identity fusion | <= 0.16733 | <= 0.51449 | <= 0.18169 | <= 0.43255 | geometry/identity trade-off |

The query-mask head does reduce fragmentation, and the region fusion raises
instance F1, but neither raises class-wise Room IoU enough to clear the PC-02
gate. Adjacency-only count enforcement and nearest-component merging also fail:
their component grouping changes class evidence and lowers Room IoU.

## Diagnosis

The residual support geometry is already strong, but footprint-only inference
does not contain enough evidence to assign interior room semantics reliably.
The remaining gap is not solved by a larger identity head alone. Two viable
next directions are:

1. use a raster-conditioned reconstruction control branch, which is legitimate
   for CubiCasa reconstruction and matches how Raster2Seq consumes floorplan
   images;
2. keep the contour-only path as a separate, weaker generation track and do not
   use it for reconstruction SOTA.

## Decision

The PC-02 query-head route is rejected for promotion. The existing DMD-v11 +
room-segmenter stack remains frozen. The next screen will add a non-room raster
control encoder using wall, opening, furniture, structure, and footprint
evidence, while explicitly excluding room-class pixels from its input.

This is a track change, not a target-control leak: the raster itself is the
reconstruction input. Any resulting row will be labeled
`raster_conditioned_reconstruction`, never `contour-only`.

The full val491 query-mask and dense-residual reports are retained in
`predicted_room_identity_full_val491_overlay_v1.json` and
`predicted_room_identity_dense_full_val491_overlay_v1.json`. Neither clears the
PC-02 gate, so those modules remain experimental and are not part of the main
pipeline.
