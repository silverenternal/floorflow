# Validation Room-Graph Refinement

Fixed evaluator: `type_geometry_hungarian`; selected candidate: `tiled_purity_050`.

| Readout | Node F1 | Edge overlap | Unit GED | Exact GED rate |
|---|---:|---:|---:|---:|
| wall_wall_primary | 0.447853 | 0.359382 | 22.578556 | 0.7335 |
| direct_tiled | 0.511164 | 0.334795 | 10.740977 | 0.9151 |
| tiled_purity_050 | 0.521297 | 0.334522 | 10.084926 | 0.9246 |
| tiled_purity_050_contact_02 | 0.521297 | 0.292552 | 9.954352 | 0.8949 |
| tiled_purity_050_contact_04 | 0.521297 | 0.286115 | 9.893843 | 0.8885 |
| tiled_purity_050_contact_08 | 0.521297 | 0.267561 | 9.830149 | 0.8790 |

| Paired delta vs baseline | Node F1 95% CI | Edge overlap 95% CI | GED 95% CI |
|---|---:|---:|---:|
| direct_tiled | +0.063311 [+0.046805, +0.079611] | -0.024586 [-0.046654, -0.003736] | -11.837580 [-12.660350, -11.029671] |
| tiled_purity_050 | +0.073444 [+0.056449, +0.090293] | -0.024860 [-0.046867, -0.004022] | -12.493631 [-13.321656, -11.688907] |
| tiled_purity_050_contact_02 | +0.073444 [+0.056449, +0.090293] | -0.066830 [-0.087968, -0.046358] | -12.624204 [-13.445886, -11.807829] |
| tiled_purity_050_contact_04 | +0.073444 [+0.056449, +0.090293] | -0.073267 [-0.094387, -0.052648] | -12.684713 [-13.505308, -11.868312] |
| tiled_purity_050_contact_08 | +0.073444 [+0.056449, +0.090293] | -0.091821 [-0.112971, -0.071847] | -12.748408 [-13.573275, -11.925690] |

The refinement changes only the graph readout. Raster labels, controls, edit masks, and outside exactness are unchanged.
