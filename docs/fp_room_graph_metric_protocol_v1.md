# Room-Graph Metric Protocol and Error Audit v1

## Decision

The canonical semantic protocol is label-plus-IoU on directly extracted,
footprint-tiled room polygons:

- one-to-one assignment;
- room-type equality is required;
- a matched node requires IoU at least 0.5;
- prediction and target use the same nearest-room footprint tiling and
  connected-component polygon extraction before matching;
- adjacency edge overlap and GED decomposition use the resulting mapping.

The type-agnostic geometry assignment and the type-plus-geometry assignment are
sensitivity diagnostics. They are not substitutes for the primary number and
must not be selected after seeing the results.

## Matching sensitivity

Seed 19, 2,479 edited CubiCasa rows:

| Protocol | Node F1 | Edge overlap |
|---|---:|---:|
| Label + IoU, primary | 0.391224 | 0.333890 |
| Geometry-only Hungarian | 0.435366 | 0.399064 |
| Type + geometry Hungarian | 0.435341 | 0.398988 |

Changing the assignment improves Node F1 by 0.044 and edge overlap by 0.065,
but both remain far below a reconstruction-grade graph score. Matching is a
measurable contributor, not the main explanation for the weak result.

The larger defect was the graph readout itself. The wall-face plan compressed
the target to 5.606 nodes per case, while direct room polygons produced 8.303
nodes. Applying the same footprint tiling and direct polygon extraction to
prediction and target changes seed-19 Node F1 from `0.3912` to `0.8371`
(paired delta `+0.4459`, 95% CI `[+0.4354, +0.4562]`) and edge overlap from
`0.3339` to `0.6578` (paired delta `+0.3239`, 95% CI `[+0.3077, +0.3396]`).
All underlying pixels and labels are identical; this is a metric/readout
correction, not a model improvement.

## Error localization

| Error | Inside edit | Boundary 1/2/5 | Far outside |
|---|---:|---:|---:|
| Node false positive | 4,009 | 1,239 | 15,022 |
| Node false negative | 1,342 | 373 | 3,524 |
| Edge false positive | 21 | 4 | 75 |
| Edge false negative | 2,890 | 868 | 9,398 |

The 1/2/5-pixel bands are mutually exclusive Chebyshev rings outside the declared
edit rectangle. Errors are not concentrated at the local-edit seam. Almost all
node and edge errors are outside the edit region, even though outside pixels and
room labels are exactly preserved. This separates pixel locality from graph
readout identity: the raster is unchanged outside, but raster-to-graph
instances and adjacencies are difficult to align with the reference graph.

## Complexity and tails

Spearman correlation between primary Node F1 and target complexity is positive
(`rho = 0.405`), not negative. Node F1 is lower on the simplest plans. Edge
overlap and GED show weak complexity correlations (`rho = 0.018` and `0.061`).
The worst 10% of cases contribute only `15.7%` of total node loss and `19.2%` of
GED, so a small long tail does not explain the aggregate.

## Ambiguity

The target graph contains same-type adjacent edges in 1,132 cases and at least
one low-purity room node in 1,955 cases. Mean target label purity is 0.896.
Reference ambiguity therefore affects the semantic ceiling, but it does not
erase the gap: the target also contains 187 undefined nodes and the prediction
contains 9,989.

## GED decomposition

The unit-cost GED mean is `20.335`. It is exact for 1,971 rows; 508 rows use an
explicit fallback because of the 24-node budget or the 15-second exact-solve
limit. Component means are:

| Component | Mean edits |
|---|---:|
| Node deletion | 6.161 |
| Node insertion | 0.098 |
| Node substitution | 1.698 |
| Edge deletion | 10.910 |
| Edge insertion | 1.467 |

GED is dominated by deletion-side node and edge edits. The next experiment should
therefore test identity alignment and adjacency construction before adding a
global pixel or loss-weight change.

## Publication boundary

These metrics remain appendix diagnostics. The result does not support a
room-graph SOTA claim. Any future promotion requires the v24 metric-governance
gate: frozen primary protocol, multi-seed improvement, paired uncertainty,
stable trend, and no regression in the main edit-detection or locality claims.

## Artifacts

- `outputs/reports/fp_room_graph_diagnostic_rows_v1.json`
- `outputs/reports/fp_room_graph_matching_protocol_v1.json`
- `outputs/reports/fp_structural_error_strata_v1.json`
- `paper/tables/fp_room_graph_matching_protocol.md`
- `paper/tables/fp_structural_error_strata.md`
