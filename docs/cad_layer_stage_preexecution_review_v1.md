# CAD Layer and Production-Stage Vector Scaffold Review

## Decision

The existing learned vector scaffold is rejected as an end-to-end CAD result.
Its geometry-aware proxy improves honest endpoint matching, but the composed
plan gate remains far below the direct room-boundary scaffold:

| Arm | Wall F1 | Opening F1 | Validity |
| --- | ---: | ---: | ---: |
| Learned free wall/opening queries | 0.0099 | 0.0037 | 0.0469 |
| Direct raster-layer + room-boundary readout | 0.3396 | 0.1183 | 1.0000 |

The comparison isolates the failure: free vector queries do not consume the
same production dependencies as a CAD plan. Interior walls are derived from
room partition boundaries, junction endpoints constrain wall geometry, and an
opening is an interval on a named host wall.

## Source Basis

The local vendored fpvec-lab paper and `fpseq` codec provide the immediate source
basis. Their wall-first contract orders walls, room-wall references, and
host-wall opening intervals into three dependent sections. The paper also
reports that an opening heatmap that the sequence decoder never consumed can be
read as host-wall intervals, and that room-centric and wall-first outputs expose
different topology failure modes.

The online literature check was attempted on 2026-10-30, but both the configured
proxy and direct access timed out. The next architecture review must retry the
2025-2026 literature check before moving beyond this local primary source.

Existing project code to reuse:

- `VectorPrimitiveHead` for queries and continuous coordinates.
- `CadActionGraph` for ordered stage tokens.
- `cad_layers.CadLayeredOutput` for independent room/wall/opening layers.
- `vector_geometry_dataset.VectorGeometryDataset` for canonical vector targets.
- `wall_graph_readout`, `opening_readout`, and `fpeval_bridge` for the fixed
  end-to-end evaluation.

## Hypothesis

Changing only the target extraction and query dependencies should recover the
gap without adding a new backbone:

1. Train wall/opening primitives against canonical cached vector geometry rather
   than re-vectorizing the raster target.
2. Decode rooms first, append room tokens to wall-decoder memory, then decode
   walls, then append wall tokens to opening-decoder memory.
3. Predict an explicit opening host-wall distribution and normalized interval
   along that wall.
4. Increase the room-query budget from four to twenty-four; the current budget
   is below the average target room count.

## Matched Control

The control is the completed geometry-aware free-query screen with the same
predicted controls, layer checkpoint, train256/cal64 split, seed 2905, eight
epochs, learning rate, and frozen room layer. The treatment changes only target
provenance and cross-stage conditioning.

Both arms must pass through `evaluate_vector_topology_scaffold_v1.py` and the
same fpeval target/extractor. Proxy wall/opening F1 alone is not a promotion
gate.

## Budget

- Train cases: 256.
- Calibration cases: 64.
- Epochs: 8 trend screen; extend to 24 only if wall, opening, or validity keeps
  improving over the final three checkpoints.
- Batch size: 4.
- Optimizer updates: about 512 for the screen.
- Additional parameters: one host-wall classifier and concatenation-only stage
  context; no new backbone.
- GPU: `cuda:0`.

## Gates

The screen advances only if:

- full-plan wall F1 exceeds the direct room-boundary scaffold `0.3396`;
- full-plan opening F1 exceeds `0.1183`;
- validity is at least `0.95`;
- room F1 does not regress below `0.2248`;
- wall/opening proxy metrics are reported separately and never substituted for
  the full-plan metrics.

## Failure Attribution

If proxy metrics improve without the plan gate:

- inspect whether host-wall predictions collapse to a single wall;
- compare canonical geometry targets with fpeval's raster-derived targets;
- measure fragmented-wall and junction-degree errors;
- compare predicted wall count with target wall count;
- only then change the vector decoder or add a reconciliation stage.

