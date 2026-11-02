# Typed Mechanism Pareto

| Arm | Violation rate | Delta-IoU | Node F1 | Edge F1 | GED | Frontier |
|---|---:|---:|---:|---:|---:|---|
| Typed | 0.000000000 | 0.937473 | 0.955068 | 0.890007 | 2.120855 | yes |
| Dense | 0.000455107 | 0.887741 | 0.953797 | 0.887080 | 2.185478 | no |
| No gate | 0.013884758 | 0.853715 | 0.949057 | 0.874121 | 2.448649 | no |
| No graph | 0.000000000 | 0.888817 | 0.953819 | 0.887200 | 2.182251 | no |

Typed routing forms a joint locality-quality and locality-graph frontier. Graphless routing can preserve locality but loses edit and graph quality; no-gate trades visible quality for measurable outside leakage.
