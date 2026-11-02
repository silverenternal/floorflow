# Structure Readout and Compositor v2

## Scope

This change targets the deterministic layer after structure prediction. It is
opt-in, preserves legacy behavior, and does not change chips/FloorSet code.

## Readout Policy

`StructureReadoutPolicy` makes the previously implicit readout explicit:

- independent wall, junction, opening, and room-boundary thresholds;
- minimum junction degree for wall retention;
- `legacy`, `none`, or `boundary_gated` room seams;
- opening support radius and minimum component area;
- fixed or resolution-scaled exterior wall width.

The default seam remains legacy for old checkpoints. The typed_v2 candidate uses
a boundary-gated seam so not every room transition becomes a wall.

## Host-Aware Openings

Two paths are now exposed:

1. Component-level filtering against predicted wall support.
2. Structure opening tensors converted into foreground probability plus family
   logits, then consumed by the existing wall-interval opening reader.

The second path attaches openings to selected wall hosts rather than relying
only on local mask overlap.

## Unified Owner-Cell Plan

`owner_cell_plan` reuses `OwnerCellComplex` to derive room polygons, shared
owner boundaries, wall segments, openings, and adjacency from the same
room-owner raster. Interior owner edges are retained only when predicted
wall/opening evidence supports them; openings are projected onto the resulting
owner-host walls. It is optional in the canonical evaluator and records
`room_source="owner_cell_complex"`.

`graph_source="plan"` derives Node/Edge/GED from the same explicit room
polygons used by fpeval. The default remains the historical raster graph.

`PlanRepairAcceptanceHead` provides a fail-closed learned gate for candidate
bridge, endpoint-snap, and opening-host repairs. Its final bias rejects all
candidates at initialization, and `select_accepted_repairs` enforces a hard
per-plan edit budget.

## Reused Components

- `LayeredFloorplan` for independent room/wall/opening layers.
- `compose_known_exterior` for the known envelope.
- `partition_wall_mask` for raw room transitions.
- `OwnerCellComplex` for shared boundaries, junctions, and owner polygons.
- Wall-interval opening readers for host-aware openings.
- Existing canonical room/wall/graph evaluators.

## Verification

- Gated seams, junction degree, opening support, adaptive exterior width, and
  owner-cell source are covered by CPU tests.
- Legacy readout remains the default.
- No GPU experiment is started by this implementation.
- Non-chip tests excluding the repository-wide line-count guard: `858 passed`.
- The only repository-wide guard failure is from two pre-existing chip files
  above 600 lines, which are outside this task's allowed edit scope.
- Changed files pass Ruff and `uv build`.
