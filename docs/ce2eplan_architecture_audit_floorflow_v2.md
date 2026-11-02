# CE2EPlan Architecture Audit and FloorFlow v2 Direction

## Evidence Boundary

This audit reads the local CE2EPlan source at commit
`774e670941eeda3e6677303ff79e8d517ae0d1c0` and its officially released
`b_model300000.pt`. It does not copy CE2EPlan's implementation and it does not
use the RPLAN test split.

Paper metadata:

- Shidong Wang and Renato Pajarola,
  *Directly From Alpha to Omega: Controllable End-to-End Vector Floor Plan
  Generation*, IEEE TVCG, July 2026.
- DOI: `10.1109/TVCG.2026.3665422`.

## What CE2EPlan Actually Builds

### Input representation

- Flattened 80-point exterior boundary.
- Flattened 8-point entrance geometry.
- Both condition vectors are repeated across every room token.

### Room representation

Each of at most eight rooms is six numbers:

```text
[exists, room_category, center_x, center_y, width, height]
```

The model predicts axis-aligned boxes. Room corners, room masks, wall
connectivity, openings, and furniture are not first-class model outputs.

### Denoising network

- Six-dimensional token input.
- Boundary and entrance projections added to every token.
- Four Transformer self-attention layers.
- A graph-attention layer over fixed room tokens.
- The GAT graph is a complete graph (`build_edge` enumerates all node pairs),
  including self edges, rather than a predicted or boundary-derived topology.
- 1,000-step diffusion.
- Approximately 9.64M parameters.

### Decoding

Boxes are aligned after sampling using boundary proximity, neighbor alignment,
and post-processing. Room polygons and walls are therefore downstream of a
rectangular box representation rather than native outputs.

## Architectural Advantages We Should Not Chase

CE2EPlan has three practical strengths:

1. a very short, fixed output sequence;
2. a mature, large training budget;
3. deterministic post-processing that hides many low-level geometric defects.

Copying the six-value room token or box-then-align pipeline would cap FloorFlow
at the same structural limitations.

## FloorFlow v2 Target

FloorFlow v2 must dominate in representation and topology, not merely add
heads to the current underfit raster path.

```text
boundary polygon
  -> explicit boundary segment graph
  -> room-program set
  -> dual room-instance representation
       mask field
       variable-length polygon corners with per-vertex validity
  -> explicit room adjacency
  -> ordered CAD action graph
       rooms -> walls -> junctions -> host openings -> furniture -> validation
  -> one CanonicalFloorplan
```

### Difference from CE2EPlan

| Capability | CE2EPlan | FloorFlow v2 |
| --- | --- | --- |
| Boundary input | flattened coordinates | boundary segment graph tokens |
| Room count | fixed maximum 8 | variable set query with learned budget |
| Room geometry | center/width/height box | mask + variable vertex polygon |
| Per-vertex validity | no | yes |
| Room adjacency | inferred after boxes | native graph output |
| Walls/junctions | post-processing | typed CAD stages |
| Openings | post-processing | host-wall conditioned outputs |
| Furniture | not in main room token | typed optional layer |
| Geometric repair | box alignment | explicit audited vector operations |
| Decode | 1,000-step diffusion | direct structured decode, target NFE 1 |

## Implemented Architecture Primitives

The following pieces are now implemented and tested:

- `BoundaryEdgeGraphEncoder`: direct segment features with length, direction,
  turning sine/cosine, cumulative arc position, and entrance marker.
- Variable-vertex room targets: original corners are padded rather than
  resampled into synthetic points.
- Per-room point-validity logits and BCE supervision.
- Polygon validity-aware rendering and CAD assembly.
- Existing mask queries, MiTA routing, adjacency logits, and typed CAD action
  graph.

All new modules are opt-in until a matched architecture smoke proves them.

## Required Architecture Work Before Metrics

1. Replace the raster-derived room memory with the explicit boundary-edge graph
   as the default opt-in room/structure memory.
2. Introduce a room-program set head that predicts room count and class quotas
   independently from room geometry.
3. Train the dual room decoder jointly:
   mask Dice/BCE, variable-corner set loss, point-validity loss, adjacency BCE,
   and room-count loss.
4. Add a compact boundary encoder ablation so the generator can be parameter
   matched near 9.64M rather than capped above 11.49M by ResNet18.
5. Connect room adjacency, wall-host assignments, and junction incidence in one
   typed graph; do not infer these only after rasterization.
6. Only after architecture contract and gradient smoke should any RPLAN metric
   screen resume.

## Architecture Acceptance Gate

- Variable room count and polygons with up to 24 real vertices.
- No box-token intermediate is required for final output.
- Exact vector-to-token round trip.
- Every room/wall/junction/opening/furniture output has a declared downstream
  consumer.
- One forward pass yields one `CanonicalFloorplan`.
- Boundary-edge graph and dual-room path have finite gradients on a real batch.
- A compact configuration exists near the 9.64M external parameter anchor.

Architecture quality is the current gate; metric comparisons remain paused.

## Matched Screen Outcome

The architecture contract passed, but the first matched quality screen did not
establish a performance win. At 256/64, eight epochs, batch size two, seed 2905,
and validation-only selection, the compact full v2 model reached Room IoU
`0.2432` and instance F1 `0.0802`; the ResNet18 class-agnostic control reached
`0.3901` and `0.1388`.

Orthogonal screens attributed the loss to the compact encoder and the current
program-set/loss composition. Boundary-edge and point-validity representations
were close to the control but did not produce a stable joint gain. A mask-derived
room-query graph was added as `RoomQueryGraphRefiner`; it preserves the strict
variable-polygon/topology contract but also underperformed at the same budget.

Detailed arm results and rejection decisions are recorded in
`docs/floorflow_v2_architecture_selection_v1.md`. Do not cite the earlier
"architecture gate passed" statement as evidence of higher generation quality;
it establishes representational capability only.
