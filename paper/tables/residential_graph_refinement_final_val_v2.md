# Validation Room-Graph Refinement

Fixed evaluator: `type_geometry_hungarian`; selected candidate: `direct_tiled`.

| Readout | Node F1 | Edge overlap | Unit GED | Exact GED rate |
|---|---:|---:|---:|---:|
| wall_wall_primary | 0.539030 | 0.481048 | 17.341826 | 0.7346 |
| direct_tiled | 0.848095 | 0.698507 | 5.219745 | 0.9469 |

| Paired delta vs baseline | Node F1 95% CI | Edge overlap 95% CI | GED 95% CI |
|---|---:|---:|---:|
| direct_tiled | +0.309065 [+0.293101, +0.325385] | +0.217459 [+0.197828, +0.237055] | -12.122081 [-12.881131, -11.397001] |

The refinement changes only the graph readout. Raster labels, controls, edit masks, and outside exactness are unchanged.
