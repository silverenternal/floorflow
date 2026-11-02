# Raster2Seq Geometry Fusion Audit v1

## Official Geometry

The upstream Raster2Seq evaluator reproduces:

- Room IoU: `0.63228`
- Room F1: `0.88863`
- Corner F1: `0.59919`
- Window/door F1: `0.34390`
- room semantic F1: `0.65082` (diagnostic because direct replay does not
  reproduce the upstream class mapping exactly)

The official checkpoints and result JSONs are present locally. The geometry
itself is therefore a valid target-free anchor.

## FloorFlow Consumer Failure

FloorFlow cannot currently consume the official geometry without losing it:

| Consumer | Room IoU |
| --- | ---: |
| Official polygon evaluator | 0.63228 |
| Native dense FloorFlow | 0.28999 |
| Native instance-room residual | 0.28917 |
| Raw Raster2Seq proposal classifier | 0.02520 |
| Proposal classifier + dense fill | 0.12660 |
| Anchored proposal classifier + dense fill | 0.16501 |

Official proposal masks overlap heavily under FloorFlow's greedy exclusive
raster painting rule. Identity correction cannot compensate for geometry lost by
that rasterization contract.

## Decision

Do not learn another room mask from the predicted-control field.

The next component must be a planar CAD partition consumer that:

1. treats official polygon vertices as immutable where possible;
2. computes a deterministic planar arrangement;
3. quantifies every polygon that is clipped, removed, or assigned to a conflict;
4. reports official polygon metrics separately from exclusive-raster metrics;
5. exposes retained-polygon IDs for later identity correction and typed layer
   fusion.

## Provenance Boundary

Any result produced by this path depends on external Raster2Seq geometry and
must not be reported as a standalone FloorFlow reconstruction score.

## Identity-prior Follow-up

Room-only proposal filtering and zero-residual anchoring to the official
Raster2Seq class prior were tested. Type accuracy reached `0.34635`, but the
exclusive-raster consumer still produced only `0.16501` Room IoU. The official
polygon evaluator remains `0.63228`. The loss occurs at the polygon-to-raster
consumer boundary, not in the identity classifier alone.

Decision: stop external raster fusion experiments. Keep Raster2Seq as an
external polygon baseline and move capability work to a native vector-first
representation with an explicit planar partition contract.
