# Predicted-control to CAD-action mapping

## Design sequence

The predicted-control protocol is mapped to the following CAD design actions:

| CAD design action | Existing model mapping | Current status |
| --- | --- | --- |
| Exterior envelope / footprint | `footprint`, contour and support controls | input exists; not a learned stage |
| Room program / schedule | `program` vector | consumed as a flat field; no program-to-room token routing |
| Room partition and instances | `room_logits`, `RasterInstanceControl`, `RoomMaskDecoder` | strongest existing room path |
| Wall segments and junctions | `structure` wall channel, `CoarseStructureRefiner`, `GeometryHeatmapHeads` | wall map exists; no explicit junction state in the channel model |
| Openings on host walls | `structure` opening/type channels, `WallIntervalHead`, opening detector | detector exists; host assignment is still partially post-hoc |
| Furniture and fixtures | furniture-control channels, `DynamicFurnitureInstanceHead` | channels exist; room-conditioned furniture queries are not part of the unified model |
| Constraint validation | validity, watertightness, `ConstraintStateEncoder`, `soft_graph_edit_distance` | evaluators and encoders exist; not used as generation-stage state |
| Edit / revision | typed edit graph and strict-locality composition | explicit graph exists, but currently post-hoc |

## Main gap

The current flat all-channel predictor receives the right predicted evidence but
does not represent the causal order of CAD design. It predicts semantic channels
in parallel. It also does not expose intermediates such as room instances, wall
junctions, host-wall attachments, or furniture containment as first-class states.

## Proposed typed action graph

Each design stage gets a token set:

- `envelope`
- `program`
- `room instances`
- `wall segments`
- `junctions`
- `openings`
- `furniture`
- `constraint/edit state`

The non-uniform links are directional:

```text
envelope -> rooms
program -> rooms
rooms <-> adjacent rooms
rooms -> boundary walls
walls -> wall junctions
walls -> openings
rooms -> openings (access)
rooms -> furniture
constraint/edit -> all active stages
```

The links are learned residual biases over a structural prior. They are not hard
masks. Hard validity and watertightness remain external constraints.

## Stage handoffs

1. Envelope and program produce room-instance queries.
2. Room instances produce boundary candidates and adjacent-room edges.
3. Wall queries consume boundary candidates and produce wall/junction state.
4. Opening queries consume wall host tokens and predict intervals and families.
5. Furniture queries consume room-instance tokens and semantic class information.
6. Constraint tokens audit the resulting graph and decide whether a stage needs
   residual repair.

Every handoff must be serializable and auditable. The model must expose
intermediate tokens/masks so failures can be attributed to partitioning, walling,
opening attachment, furniture placement, or validation.

## Reuse boundary

Reuse the existing `RoomMaskDecoder`, `WallIntervalHead`,
`DynamicFurnitureInstanceHead`, `GraphEditEncoder`, `ConstraintStateEncoder`,
`soft_graph_edit_distance`, and `typed_channel_graph`. New code should provide
the typed stage interface and handoff contract, not duplicate those modules.

## First staged implementation result

The room stage now consumes the raster-instance prediction rather than the weak
cached room logits. Under the matched train256/cal64 eight-epoch screen this
stabilizes Room IoU around `0.308`, with instance F1 around `0.22`.

The all-channel residual still does not replace specialized structure heads:
wall F1 remained around `0.06` and furniture F1 below `0.01`. The next iteration
must route the canonical structure bridge and centerline wall evidence into the
wall/junction stage, and the host-wall interval detector into the opening stage.

The following staged run connected those canonical heads and added independent
wall/opening/furniture losses. Room IoU remained stable at `0.308`; opening F1
rose from roughly `0.05` to `0.099` across the eight checkpoints. Wall and
furniture did not improve because they were still decoded from the shared
65-channel softmax. The next implementation must give each stage its own output
head and use the shared channel graph only as a routing mechanism.

## Explicit layer contract

`CadLayeredOutput` now exposes independent room, wall, opening, furniture, and
support layers. Each layer has its own logits, threshold, loss, and F1. The
8-epoch screen produced:

- Room IoU `0.3121`
- Wall F1 `0.5425`
- Furniture F1 `0.0971`
- Opening F1 `0.0528`
- Instance F1 `0.2083`

This confirms that layer separation fixes wall/furniture decoding, while opening
must remain a vector interval problem rather than a raster class map.

## Layer-constrained evaluation

The layer renderer now enforces two CAD relations: opening pixels must lie in a
small wall band, and furniture pixels must lie inside a room. Room prediction is
kept from the validated raster-instance stage.

Independent calibration64 evaluation of the best staged checkpoint:

| Metric | Value |
| --- | ---: |
| Room IoU | 0.3081 |
| Instance F1 | 0.2198 |
| Room layer F1 | 0.8703 |
| Wall layer F1 | 0.5397 |
| Opening layer F1 | 0.2192 |
| Furniture layer F1 | 0.0981 |
| HouseMind-style Node F1 | 0.7304 |
| HouseMind-style Edge F1 | 0.4838 |
| HouseMind-style GED | 8.625 |

The remaining failure is not layer classification. It is vectorization and
topology: wall area/continuity and opening host intervals do not yet reconstruct
the target CAD graph robustly. The next implementation should make wall and
opening layers vector-first.

The first vector-first probe used the raster layers as inputs to the existing
skeleton/segment readout. It reached vector Wall F1 `0.3279`, Opening F1
`0.0618`, and Room F1 `0.2248`, well below the layer-level scores. This is a
controlled negative result: direct raster-to-vector conversion is the bottleneck,
so the next model must predict the vector topology scaffold itself rather than
post-process the layer masks.

Adding the room-partition boundary as a vector scaffold raised Wall F1 to
`0.3392` and Opening F1 to `0.1170`, but Room F1 remained `0.2248`. The scaffold
helps host-wall and interval decoding, yet room instance reconstruction still
needs a learned instance/topology stage.

The learned vector scaffold, which consumes action tokens and predicts wall
endpoints plus opening centers/width/type directly, reached vector Wall F1
`0.4297` and Opening F1 `0.4145` at eight epochs. This is a clear improvement
over skeleton conversion and validates the vector-first direction. Those
detection metrics are not yet final CAD metrics; the next gate composes the
scaffold with the frozen room layer and runs fpeval/watertightness.

The full-plan gate exposed a matching flaw: center-only Hungarian matching made
the scaffold endpoint score look strong, but composed fpeval reached only Wall
F1 `0.0116`, Opening F1 `0.0039`, and validity `0.0156`. The next scaffold loss
must match ordered endpoints with orientation and length costs, and openings
must be associated to host walls before they are considered valid primitives.

The geometry-aware correction produced honest endpoint matching, but did not
fix the composed plan: its eight-epoch calibration64 gate reached Wall F1
`0.0099`, Opening F1 `0.0037`, and validity `0.0469`. The direct room-boundary
scaffold remains substantially stronger at Wall F1 `0.3396`, Opening F1
`0.1183`, and validity `1.0`.

This isolates the remaining defect to the production-stage contract. The free
wall queries do not consume room-partition boundaries, and the free opening
queries do not condition on a named wall host. The next screen therefore keeps
the same queries and layer checkpoint but changes target provenance and stage
dependencies:

1. Train against the canonical vector geometry cache rather than targets
   re-extracted from the raster label map.
2. Decode rooms first, inject their tokens into wall-decoder memory, then inject
   wall tokens into opening-decoder memory.
3. Predict an opening host-wall distribution and interval along the selected
   host instead of decoding an unrelated absolute center.
4. Raise the room-query budget from four to twenty-four.

The pre-execution review is recorded in
`docs/cad_layer_stage_preexecution_review_v1.md`.

The learned free-query treatment still failed the composed gate: Wall F1
`0.0030`, Opening F1 `0.0219`, and validity `0.0469`. The canonical target was
verified against the fpeval raster target using the correct 384 frame, where it
reached Wall F1 `0.5278` and Opening F1 `0.9011`; therefore the failure is
endpoint localization, not target provenance.

The zero-training production-layer alternative now passes the
target-watertight calibration subset:

| Eligible metric | Value |
| --- | ---: |
| Wall F1 | 0.3683 |
| Opening F1 | 0.1211 |
| Room F1 | 0.2322 |
| Schema validity | 1.0000 |
| Watertightness | 1.0000 |
| Node F1 | 0.7188 |
| Edge F1 | 0.4714 |
| GED | 7.9091 |

Accepted composition:

```text
room cells
  -> junction-split room/footprint wall candidates
  -> staged wall-layer evidence selects candidates
  -> high-recall runtime skeleton supplies opening host intervals
  -> dedicated opening detector attaches intervals to selected walls
```

The next treatment replaces only the deterministic candidate threshold with a
learned scorer and bounded endpoint residual. Its matched control and budget are
recorded in `docs/cad_wall_candidate_preexecution_review_v2.md`.
