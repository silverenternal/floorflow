# Validation Room-Graph Refinement

Fixed evaluator: `type_geometry_hungarian`; selected candidate: `tiled_no_undefined_purity_050`.

| Readout | Node F1 | Edge overlap | Unit GED | Exact GED rate |
|---|---:|---:|---:|---:|
| wall_wall_primary | 0.447853 | 0.359382 | 22.569002 | 0.7378 |
| direct_tiled | 0.511164 | 0.334795 | 10.749469 | 0.9119 |
| tiled_no_undefined | 0.511164 | 0.334795 | 10.745223 | 0.9108 |
| tiled_purity_050 | 0.521297 | 0.334522 | 10.092357 | 0.9204 |
| tiled_no_undefined_purity_050 | 0.521297 | 0.334522 | 10.091295 | 0.9236 |

| Paired delta vs baseline | Node F1 95% CI | Edge overlap 95% CI | GED 95% CI |
|---|---:|---:|---:|
| direct_tiled | +0.063311 [+0.046805, +0.079611] | -0.024586 [-0.046654, -0.003736] | -11.819533 [-12.634820, -11.004246] |
| tiled_no_undefined | +0.063311 [+0.046805, +0.079611] | -0.024586 [-0.046654, -0.003736] | -11.823779 [-12.639066, -11.008493] |
| tiled_purity_050 | +0.073444 [+0.056449, +0.090293] | -0.024860 [-0.046867, -0.004022] | -12.476645 [-13.306821, -11.675133] |
| tiled_no_undefined_purity_050 | +0.073444 [+0.056449, +0.090293] | -0.024860 [-0.046867, -0.004022] | -12.477707 [-13.306821, -11.675133] |

The refinement changes only the graph readout. Raster labels, controls, edit masks, and outside exactness are unchanged.
