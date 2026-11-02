# Canonical Plan and Token Optimization v2

## Scope

This change connects the explicit `RasterWallPlan` to the canonical CAD and
unified token layers without re-deriving rooms or walls. It is CPU-validated
and does not modify chips/FloorSet code.

## Lossless Plan Adapter

`raster_plan_to_canonical` preserves:

- stable owner-cell room IDs;
- explicit room and wall IDs;
- every opening with a valid host wall ID;
- owner-cell adjacency;
- observed junction positions and degrees;
- source hash and canonical operations.

It uses a resolution-adaptive coordinate quantizer and validates the result with
the existing canonical grammar.

## Canonical Assembly Defaults

`assemble_vector_floorplan` now selects `owner_cells` automatically when an
owner cell complex is available. The room-polygon wall projection remains a
fallback for legacy callers.

The owner-cell branch consumes shared owner 1-cells directly. It no longer
reconstructs walls from room polygons.

Openings and typed junctions can be supplied explicitly and remain attached to
their canonical wall hosts. The previous entrance-only behavior is retained as
a fallback.

## Token Round-Trip

`canonical_token_roundtrip` reports:

- canonical fingerprint equality;
- room/wall/opening count equality;
- adjacency equality;
- junction degree equality;
- token count and byte count;
- maximum quantization error.

The canonical evaluator exposes this through
`--canonical-token-readout`, and the typed_v2 candidate enables it.

## Adaptive Geometry Contracts

- Coordinate quantization scales with image resolution.
- Maximum wall-gap closure scales with the plan frame instead of using a fixed
  30-pixel cap.
- `graph_source=plan` keeps fpeval and graph metrics on the same room polygons.

## Reused Components

- Existing `RasterWallPlan`, `CanonicalFloorplan`, and canonical validators.
- Existing unified tokenizer and strict decoder.
- Existing `OwnerCellComplex` and shared-boundary wall candidates.
- Existing `plan_repair_acceptance` and structural relation proxies.

The training-side proxy is exposed as `canonical_plan_proxy_loss` and reuses
the existing room-edge, opening-host, junction, and room-boundary relation
terms. It does not attempt to differentiate through token strings.

## Claim Boundary

No GPU metric result is produced by this implementation. Promotion still
requires the canonical round-trip and plan-source graph to pass the experiment
gates on calibration and disjoint holdout.

## Verification

- Canonical adapter, owner-cell assembly, opening preservation, and token
  round-trip tests pass.
- Non-chip suite excluding the repository-wide chip line-count guard:
  `861 passed`.
- Changed files pass Ruff and all non-chip Python files remain below 600 lines.
- `uv build` succeeds.
