# Compute-Matched Typed vs No-Graph

| Quantity | Value |
|---|---:|
| Added learned parameters | 0 |
| Network NFE change | 0 |
| Typed latency | 3.192470 ms |
| No-graph latency | 0.266605 ms |
| Typed overhead | 2.925865 ms |

| Metric | Typed - budget-matched No-graph | 95% bootstrap CI |
|---|---:|---:|
| outside_pixel_exact | +0.000000 | [+0.000000, +0.000000] |
| locality_violation_rate | +0.000000 | [+0.000000, +0.000000] |
| delta_iou | +0.048656 | [+0.045549, +0.051844] |
| node_f1 | +0.001249 | [+0.000898, +0.001614] |
| edge_f1 | +0.002808 | [+0.002365, +0.003284] |
| ged | -0.061396 | [-0.070432, -0.052844] |

This is a post-hoc routing compute audit, not evidence that Typed is more efficient. Typed consumes more routing time and is not claimed as an efficiency contribution.
