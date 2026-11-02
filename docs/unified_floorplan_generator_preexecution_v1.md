# Unified Floorplan Generator Preexecution Review v1

## Scope

This review starts SOTA-04 on the RPLAN boundary-only generation track. It uses
the local CE2EPlan checkout and its authorized preprocessed RPLAN JSON. It does
not inspect CE2EPlan test predictions or RPLAN test data, and it does not modify
the chips/FloorSet path.

## Local Data Contract

The CE2EPlan checkout is commit
`774e670941eeda3e6677303ff79e8d517ae0d1c0`, which matches public `HEAD` on
2026-10-30. Its release asset is `Trained_model`, published 2026-03-14.

The local preprocessed RPLAN fixture contains:

| Split | Cases | Used now |
| --- | ---: | --- |
| train | 56,053 | yes |
| val | 12,018 | yes |
| test | 12,002 | no |

Fields include exterior boundary, entrance rectangle, room category, room
polygon/box, and room adjacency. A 2,000-case train audit and a 2,000-case val
audit found:

- room count range 4-8;
- room vertex range 4-24;
- boundary vertex range 5-36;
- no invalid boundary, invalid room polygon, overlap, out-of-boundary room, or
  invalid adjacency edge in the audited prefixes;
- derived room and exterior boundary segments have p95 64 and maximum below 100
  in each audited prefix.

RPLAN does not provide interior door/window geometry or furniture instances.
The RPLAN generator therefore emits a CAD scaffold with rooms, walls, junctions,
support, adjacency, and the supplied entrance opening. Furniture and interior
openings remain first-class optional outputs in the schema but are not claimed as
supervised RPLAN results.

## Reused Components

- `BoundaryGeometryEncoder` converts the footprint contour into ordered geometric
  tokens.
- `ContourRoomSegmenter` provides a reusable boundary-conditioned spatial encoder.
- `RoomMaskDecoder` provides room-instance identity queries and set supervision.
- `OfficialMiTARouteAttention` routes room identity queries to boundary tokens
  with sparse top-k landmark routing.
- `VectorPrimitiveHead` provides room polygon, wall, and host-conditioned opening
  query heads.
- `polygon_loop_loss` supplies Hungarian room matching plus polygon, Chamfer,
  mask, and query-validity losses.
- `room_mask_loss` supplies identity-mask and class supervision.
- `room_boundary_candidates` converts decoded room polygons into exact line
  segments before junction derivation.

## Architecture

```text
boundary points
  -> contour geometry tokens + boundary raster
  -> spatial room encoder
       -> room-instance identity queries
       -> MiTA boundary-to-room routing
  -> vector primitive query decoder
       -> room class and polygon
       -> wall candidates and types
       -> entrance host-wall interval
  -> deterministic CAD assembler
       -> rooms
       -> walls and junctions
       -> support/footprint
       -> adjacency
       -> explicit decode operations
  -> CanonicalFloorplan and unified token codec
```

Room identity and room geometry are not flattened into one feature vector:
identity queries first attend to the boundary memory, then condition the polygon
and structure queries through the vector decoder. Stage ownership is explicit:
envelope and support are boundary-derived; rooms own identity and polygons;
walls/junctions consume rooms; openings consume host walls; validation consumes
the fully assembled vector object.

## Supervision

The first mechanism screen uses:

- Hungarian identity class and mask losses;
- Hungarian polygon L1, Chamfer, soft-mask Dice, and query-existence losses;
- wall-candidate Hungarian endpoint/existence/type losses;
- entrance center, width, and host-wall losses;
- room-count supervision;
- optional program/adjacency conditioning as a declared ablation.

The wall layer is also audited against deterministic room-boundary candidates so
a learned wall query head cannot silently contradict the decoded room partition.

## Iteration Budget

1. Contract smoke: one batch forward, every declared loss, backward, finite
   gradients, checkpoint reload, vector assembly, token round-trip, and validator.
2. Gradient and shape smoke on 64 train cases / 32 validation cases.
3. Eight-epoch 256/64 matched architecture screen.
4. Promote only a positive final-three-checkpoint mechanism.

All GPU work uses `cuda:0`. Validation selects checkpoints; RPLAN test data stays
closed.

## Initial Gates

- No NaNs and no disconnected declared heads.
- Room mask, polygon, wall, opening, and count losses all contribute finite
  gradients.
- At least one decoded plan per validation batch passes the vector validator.
- Token round-trip equals the assembled canonical object.
- The deterministic boundary-derived control is always reported alongside the
  learned room path.

Quality gates from `fptodo.json` remain unchanged. This review does not relax
FID, MSE, Node F1, edge overlap, GED, validity, or watertightness to make the
first underfit run look successful.
