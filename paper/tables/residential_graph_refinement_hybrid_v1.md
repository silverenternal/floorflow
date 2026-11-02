# Validation Room-Graph Refinement

Fixed evaluator: `type_geometry_hungarian`; selected candidate: `tiled_purity_050_wall_union`.

| Readout | Node F1 | Edge overlap | Unit GED | Exact GED rate |
|---|---:|---:|---:|---:|
| wall_wall_primary | 0.447853 | 0.359382 | 22.552017 | 0.7325 |
| direct_tiled | 0.511164 | 0.334795 | 10.740977 | 0.9151 |
| tiled_purity_050 | 0.521297 | 0.334522 | 10.089172 | 0.9278 |
| tiled_purity_050_wall_edges | 0.521297 | 0.314085 | 9.875796 | 0.9161 |
| tiled_purity_050_wall_union | 0.521297 | 0.341735 | 10.349257 | 0.9299 |

| Paired delta vs baseline | Node F1 95% CI | Edge overlap 95% CI | GED 95% CI |
|---|---:|---:|---:|
| direct_tiled | +0.063311 [+0.046805, +0.079611] | -0.024586 [-0.046654, -0.003736] | -11.811040 [-12.633785, -11.009554] |
| tiled_purity_050 | +0.073444 [+0.056449, +0.090293] | -0.024860 [-0.046867, -0.004022] | -12.462845 [-13.289862, -11.664490] |
| tiled_purity_050_wall_edges | +0.073444 [+0.056449, +0.090293] | -0.045297 [-0.065851, -0.025233] | -12.676221 [-13.508493, -11.871523] |
| tiled_purity_050_wall_union | +0.073444 [+0.056449, +0.090293] | -0.017646 [-0.039736, +0.003536] | -12.202760 [-13.035032, -11.400212] |

The refinement changes only the graph readout. Raster labels, controls, edit masks, and outside exactness are unchanged.
