# Dual Pareto Evidence v2

Model NFE is fixed across typed-routing points. The second x-axis measures deterministic post-processing latency, not model sampling.

## Quality-Locality

| Method | Outside exact | Delta-IoU | Frontier |
|---|---:|---:|---|
| FloorFlow control-gated | 1.000000 | 0.987754 | yes |
| FloorFlow raw renderer | 0.989987 | 0.866409 | no |
| typed_edit_graph | 1.000000 | 0.939087 | no |
| dense_cross_attention | 0.999680 | 0.897817 | no |
| no_edit_gating | 0.989987 | 0.866409 | no |
| no_structural_graph | 1.000000 | 0.898552 | no |
| Raster2Seq native | 0.920721 | 0.096246 | no |
| Raster2Seq outside-frozen redraw | 1.000000 | 0.163395 | no |

## Safety-Efficiency

| Method | Routing ms/case | Verifiability | Frontier |
|---|---:|---:|---|
| typed_edit_graph | 3.579 | 0.963292 | no |
| dense_cross_attention | 1.161 | 0.963002 | no |
| no_edit_gating | 0.301 | 0.942469 | no |
| no_structural_graph | 0.286 | 0.963695 | yes |

Raster2Seq rows are reconstruction/editing-wrapper comparisons and are excluded from the safety-efficiency panel because the required verifiability factors are not jointly reported.
