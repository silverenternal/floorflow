# Validation Room-Graph Refinement

Fixed evaluator: `type_geometry_hungarian`; selected candidate: `direct_tiled`.

| Readout | Node F1 | Edge overlap | Unit GED | Exact GED rate |
|---|---:|---:|---:|---:|
| wall_wall_primary | 0.539030 | 0.481048 | 17.875796 | 0.6507 |
| direct_tiled | 0.848095 | 0.698507 | 5.345011 | 0.9140 |
| tiled_purity_050 | 0.845787 | 0.682526 | 5.648620 | 0.9151 |
| tiled_purity_050_wall_edges | 0.845787 | 0.614846 | 6.697452 | 0.8896 |
| tiled_purity_050_wall_union | 0.845787 | 0.673214 | 5.900212 | 0.8896 |

| Paired delta vs baseline | Node F1 95% CI | Edge overlap 95% CI | GED 95% CI |
|---|---:|---:|---:|
| direct_tiled | +0.309065 [+0.293101, +0.325385] | +0.217459 [+0.197828, +0.237055] | -12.530786 [-13.310005, -11.783413] |
| tiled_purity_050 | +0.306757 [+0.290818, +0.322962] | +0.201477 [+0.182267, +0.220658] | -12.227176 [-12.996842, -11.474496] |
| tiled_purity_050_wall_edges | +0.306757 [+0.290818, +0.322962] | +0.133798 [+0.116299, +0.151421] | -11.178344 [-11.955467, -10.409660] |
| tiled_purity_050_wall_union | +0.306757 [+0.290818, +0.322962] | +0.192166 [+0.173338, +0.211321] | -11.975584 [-12.736757, -11.210111] |

The refinement changes only the graph readout. Raster labels, controls, edit masks, and outside exactness are unchanged.
