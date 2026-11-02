# Predicted-Control Generation Plausibility v2

## Scope

This phase evaluates whether target-free predicted-control outputs form a
reasonable floorplan before optimizing Room IoU. It uses CubiCasa validation
only: 64 calibration cases at offset 0 and 64 disjoint holdout cases at offset
128. Test503 remains closed. Target labels are read only for the Room IoU
diagnostic; no target geometry or room pixels are used during generation.

## Reused Public Implementations

| Source | Commit | Reused part | Decision |
|---|---|---|---|
| [fpvec-lab](https://github.com/Cyprinus12138/fpvec-lab) | `b4b4a169` | `fpeval.verify`, validity, wall/opening/room matching, edit cost | Direct dependency |
| [Raster2Seq](https://github.com/Cornell-VAILab/Raster2Seq) | `a6c4e27a` | Room/corner/angle metric definitions in `evaluations/s3d_floorplan_eval/Evaluator/Evaluator.py` | Local row adapter because the upstream evaluator depends on its dataset loader |
| [GSDiff](https://github.com/SizheHu/GSDiff) | `a02af021` | Typed room graph, shared-edge adjacency, and graph-distance reference | Fixed local graph evaluator |

The RoomFormer metric lineage is used as the reference for 0.5 room IoU,
10-pixel corner tolerance, and 5-degree angle tolerance. The implementation is
in `src/floorflow/evaluation/room_corner_metrics.py`.

## Frozen Definitions

### Raster plausibility

- `room_coverage`: predicted room pixels divided by habitable interior pixels.
  Habitable pixels are inside the known footprint and exclude predicted walls,
  openings, and furniture. Walls therefore cannot be counted as missing rooms.
- `room_area_fraction_of_interior`: room pixels divided by all interior pixels.
  This is reported separately as an occupancy diagnostic.
- `wall_closure`: size of the largest 8-connected wall-or-opening component
  divided by all structural pixels.
- `opening_wall_alignment`: opening pixels within two pixels of a predicted wall
  or the known exterior band, conditional on non-empty openings.
- `furniture_room_containment`: furniture pixels within a 12-pixel support
  dilation of predicted rooms, conditional on non-empty furniture.
- `outside_footprint_fraction`: non-background labels outside the footprint
  divided by all outside pixels.
- `room_component_count`: number of connected room instances across room classes.

### CAD plausibility

- Parseability, five-check validity, wall continuity, closure, overlap, and
  out-of-boundary ratios use the existing FloorFlow/FEVP implementations.
- `watertight`: the raw `fpeval.verify` room-partition check. It is kept
  unchanged for protocol compatibility.
- `room_face_count`: number of explicit room faces produced by the fixed wall
  graph readout.
- `room_face_retention`: `min(1, room_face_count / room_component_count)`.
- `functional_watertight`: raw watertightness conjoined with room-face
  retention >= 0.80, exact footprint-area coverage, and the same number of
  connected components as the input footprint. This is a local diagnostic, not
  a replacement for FEVP; raw watertightness is always reported separately.

The face-retention diagnostic is necessary because raw watertightness can pass
when many predicted room components collapse into one large wall-graph face.

### Tolerances

The FEVP contract uses room match IoU 0.5, room minimum area 4 px, room overlap
4 px2, edge contact 2 px, wall junction clustering 2 px, and hole minimum area
0.25 px. Wall/opening matching is reported at `tol_frac=0.05`. These values are
stored in `configs/fevp_metric_contract_v2.json` and
`src/floorflow/evaluation/fevp_metric_contracts.py`.

## Selected Representation

The selected prototype keeps the learned dense room partition as explicit room
instances. Each 4-connected room component is converted to an exact union of
pixel cells; nested rooms are represented as interior holes rather than being
silently filled. Predicted walls/openings are retained and a one-pixel seam is
added only in the rendered structural layer. The explicit room polygons remain
unchanged, so adding a wall cannot erase room content.

The local fpeval adapter was extended with backward-compatible polygon holes.
Without this change, a closet nested inside another room was serialized as an
overlap and incorrectly failed watertightness.

## Results

The selected readout retains the model/openings, adds a one-pixel room seam,
and applies 0.04-frame endpoint gap reconciliation to the wall graph while
preserving the explicit room polygons. The table separates calibration,
disjoint holdout, and full validation. Room IoU is diagnostic only and was not
used to select the readout.

| Split | Cases | Room IoU* | Raster plausible | Global wall closure | Componentwise wall closure | Wall area | Opening align | Face retention | Validity | Raw watertight | Functional watertight | Wall continuity | Plan plausible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Calibration | 64 | 0.2795 | 0.9844 | 0.9681 | 0.9979 | 0.0956 | 0.9959 | 0.9956 | 1.0000 | 0.8906 | 1.0000 | 0.9951 | 0.9688 |
| Holdout128 | 64 | 0.3182 | 0.9688 | 0.9560 | 0.9961 | 0.0896 | 0.9937 | 0.9978 | 1.0000 | 0.8594 | 1.0000 | 0.9952 | 1.0000 |
| Val491 | 491 | 0.2697 | 0.9857 | 0.9484 | 0.9974 | 0.0939 | 0.9951 | 0.9957 | 1.0000 | 0.8167 | 0.9980 | 0.9955 | 0.9878 |

`*` Diagnostic only; not a promotion criterion in this phase.

The full validation wall-area fraction is `0.0939`; the target floorplans have a
mean wall-area fraction of `0.0872` over the same split. The old bridge produced
only `0.0467`, so its apparently high raw watertightness was partly explained by
under-walling and room-face collapse.

The original wall-graph readout retained only `0.1650` of exportable room
instances as CAD faces on validation. The selected representation retains
`0.9957`, has exact validity/closure, zero room overlap, and zero out-of-boundary
area.

Raw fpeval watertightness is lower than the old `0.9375` because it no longer
counts a single collapsed face as a complete multipart plan. The remaining raw
failures are dominated by inputs with 2-3 disconnected footprint components,
which are counted separately by `componentwise_watertight`.

Among the 401 validation inputs with a single connected footprint, raw
watertightness and componentwise watertightness are both `1.0000`. The other 90
inputs contain 2-3 footprint components, where the standard fpeval
single-region rule reports raw watertightness `0.0000`; the componentwise
diagnostic is `0.9889` for that stratum. This distinction is a protocol
boundary, not a reason to alter the external metric.

The final five-seed audit uses the Student-t 95% interval with `n=5`:

| Metric | Mean | SD | Range | 95% CI |
|---|---:|---:|---:|---:|
| Raster plausible | 0.98493 | 0.00112 | [0.98371, 0.98574] | [0.98354, 0.98631] |
| Room-face retention | 0.99565 | 0.00000 | exact | exact |
| Functional watertightness | 0.99796 | 0.00000 | exact | exact |
| Wall continuity | 0.99543 | 0.00008 | [0.99530, 0.99551] | [0.99533, 0.99553] |
| Plan plausibility | 0.98656 | 0.00112 | [0.98574, 0.98778] | [0.98517, 0.98794] |
| Room IoU* | 0.26970 | 0.00007 | [0.26960, 0.26978] | [0.26961, 0.26979] |

All five seeds pass every frozen plausibility gate. The remaining fidelity
ceiling is Room IoU/instance identity, not structural reasonableness.

Rejected mechanisms:

- Replacing model walls with partition walls loses room content in the rendered
  output and does not survive the disjoint holdout.
- Radius-1/2 seam walls improve wall coverage but overproduce walls and do not
  improve plan-level validity consistently.
- Endpoint reconciliation and IP wall snapping do not recover missing room
  faces from the original wall graph.
- Constructing wall centerlines directly from every cell boundary produces
  hundreds of fragmented walls and wall continuity `0.87`, so it is not
  retained.

## Decision

Accept and freeze the explicit room-cell, one-pixel-seam, gap-reconciled
representation as the predicted-control plausibility baseline. It materially
fixes room identity, hole serialization, overlap, validity, componentwise
watertightness, wall continuity, and wall-area calibration without target
information.

Move to Room IoU/instance fidelity. Every later candidate must retain these
five-seed gates. Raw and functional watertightness, and global and
componentwise wall closure, must remain separate in all later reports.

Machine-readable evidence:

- `outputs/reports/predicted_control_plausibility_cal64_final_v12.json`
- `outputs/reports/predicted_control_plausibility_holdout64_offset128_final_v12.json`
- `outputs/reports/predicted_control_plausibility_val491_final_v12.json`
- `outputs/reports/predicted_control_plausibility_fiveseed_v12.json`
