# FEVP Graph Metrics v2

| Arm | Node P | Node R | Node F1 | Edge P | Edge R | Edge F1 | GED | Exact GED |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| typed_edit_graph | 0.995885 | 0.925830 | 0.955068 | 0.910891 | 0.823376 | 0.890007 | 2.120855 | 0.991206 |
| dense_cross_attention | 0.995109 | 0.924620 | 0.953797 | 0.907847 | 0.821298 | 0.887080 | 2.185478 | 0.990077 |
| no_edit_gating | 0.994090 | 0.918630 | 0.949057 | 0.896645 | 0.809425 | 0.874121 | 2.448649 | 0.980637 |
| no_structural_graph | 0.995109 | 0.924650 | 0.953819 | 0.907965 | 0.821363 | 0.887200 | 2.182251 | 0.989996 |

Typed GED 2.120855 versus no-gate 2.448649, dense 2.185478, and graphless 2.182251.
GED comparisons use paired Wilcoxon tests and a fixed evaluator.
