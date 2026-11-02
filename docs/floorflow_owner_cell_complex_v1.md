# FloorFlow Owner Cell Complex v1

## Purpose

The boundary graph and confidence-gated owner readout improved room identity and
graph consistency, but walls were still derived only after separately snapping
and clipping room polygons. This stage introduces one explicit owner cell
complex so room adjacency, shared boundaries, junctions, walls, and opening
hosts consume the same predicted topology.

No RPLAN test row is used. All evidence is matched validation evidence with one
NFE and `cuda:0`.

## Representation

`src/floorflow/geometry/owner_cell_complex.py` defines:

- room 2-cells from the selected-query owner raster;
- shared interior and support-boundary 1-cells;
- owner pair, boundary type, and confidence on each 1-cell;
- room adjacency directly from shared interior edges;
- junction 0-cells from clustered boundary endpoints;
- continuous wall polylines produced by planar line merge and staircase
  simplification.

The canonical assembler supports four wall sources:

| Source | Definition |
| --- | --- |
| `room_polygons` | Existing clipped room-polygon boundary path |
| `owner_cells` | Exact owner-cell polygons, still vectorized independently |
| `shared_graph` | Raw owner 1-cells plus bounded endpoint connectors |
| `shared_network` | Merged owner 1-cells with simplified continuous polylines |

`shared_network` is the promoted path. `shared_graph` was retained as a
diagnostic because it reached higher wall overlap with many connector repair
segments; it is not the final architectural claim.

## Engineering Fixes

The first shared-graph attempt exposed two defects:

1. Clipped room-polygon rasterization could turn a sub-pixel exterior channel
   into a false interior hole. The raster diagnostic now fills this seam only
   when the vector room union is already hole-free.
2. Some owner support regions consisted of two large islands separated by
   2.0-3.61 pixels. `bridge_fill` connects substantial components with
   one-pixel shortest bridges before canonical decoding.

The owner-network wall path also required an entrance-host correction. The
entrance center is preserved and its explicit host wall is attached to the
wall network; projecting the center onto a different nearby wall changed the
condition-derived opening metric and was rejected.

## Matched Readout Screen

Five seeds (`19`, `29`, `37`, `43`, `2905`) are evaluated on the same
4096/512 setup with validation offset 256. The `room_polygons` and
`shared_network` arms use the same checkpoint, owner partition, adaptive gap
policy, confidence, count prediction, and one NFE.

| Metric | Room polygons | Shared owner network | Delta |
| --- | ---: | ---: | ---: |
| Room IoU | 0.67587 | 0.67587 | 0.00000 |
| Instance F1 | 0.52793 | 0.52793 | 0.00000 |
| Node F1 | 0.92449 | 0.92449 | 0.00000 |
| Edge overlap | 0.62203 | 0.62203 | 0.00000 |
| GED | 3.03242 | 3.03242 | 0.00000 |
| Validity | 1.00000 | 1.00000 | 0.00000 |
| Conditional watertightness | 0.99961 | 0.99961 | 0.00000 |
| Dangling endpoint rate | 0.02168 | **0.00842** | **-0.01326** |
| Wall F1 | 0.62166 | **0.67590** | **+0.05424** |
| Wall IoU | 0.43592 | **0.47488** | **+0.03896** |
| Opening F1 | 1.00000 | 1.00000 | 0.00000 |

Wall F1 and wall IoU improve on all five seeds. Their paired 95% t-intervals
are `[0.04814, 0.06035]` and `[0.03271, 0.04521]`. The dangling endpoint rate
improves on all five seeds with interval `[-0.01454, -0.01198]`. Exact
two-sided Wilcoxon p-values are `0.0625` because there are only five paired
seeds; rank-biserial effect is `1.0`. This is evidence of consistent direction,
not a claim of `p < 0.05`.

Connector diagnostics:

- mean connectors per case: `0.2555`;
- maximum seed mean: `0.2891`;
- mean connector length: `0.6582` pixels;
- maximum connector: `0.7592` pixels;
- connector share of total wall length: `0.0781%`.

Thus at least `99.9%` of wall length is derived directly from the merged owner
cell complex, with only sub-pixel endpoint closure.

## Claim Boundary

Allowed:

- the owner cell complex directly supplies room adjacency, shared boundaries,
  junctions, and canonical wall geometry;
- on the matched five-seed RPLAN validation readout, shared owner walls improve
  wall F1 and wall IoU without changing room, graph, validity, watertightness,
  or opening metrics;
- dangling endpoint rate falls below `0.01`.

Not allowed:

- learned wall or opening SOTA;
- RPLAN test or leaderboard claims;
- complete-CAD external SOTA;
- treating the condition-derived entrance opening as a learned opening result.

Evidence:
`outputs/reports/floorflow_sharednetwork_fiveseed_512_v1.json`.
