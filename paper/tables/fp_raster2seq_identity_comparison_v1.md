# Raster2Seq Identity Adapter Diagnostic v1

This is a project-defined official-style input-adapter diagnostic. It is not upstream Raster2Seq preprocessing and is not eligible as a matched reconstruction table.

| Metric | FloorFlow | R2S adapter diagnostic | FloorFlow - adapter | Paired 95% CI |
|---|---:|---:|---:|---:|
| room_macro_iou | 0.988793 | 0.069145 | +0.919648 | [+0.903772, +0.934247] |
| instance_f1 | 0.988199 | 0.517106 | +0.471092 | [+0.449934, +0.491893] |
| instance_type_accuracy | 1.000000 | 0.163056 | +0.836944 | [+0.812478, +0.860303] |
| room_support_iou | 0.996749 | 0.629246 | +0.367503 | [+0.350026, +0.384465] |
| node_f1 | 1.000000 | 0.304735 | +0.695265 | [+0.676527, +0.714227] |
| edge_overlap | 0.992656 | 0.055074 | +0.937582 | [+0.920098, +0.953805] |
| ged | 0.095427 | 14.506958 | -14.411531 | [-15.204871, -13.618241] |
| exact_ged_rate | 1.000000 | 1.000000 | +0.000000 | [+0.000000, +0.000000] |
| wall_f1 | 0.971102 | 0.000000 | +0.971102 | [+0.969315, +0.972806] |
| opening_f1 | 0.986420 | 0.087502 | +0.898917 | [+0.890882, +0.906450] |
| furniture_f1 | 0.991145 | 0.015905 | +0.975241 | [+0.963482, +0.985077] |
| pixel_accuracy | 0.987812 | 0.087015 | +0.900797 | [+0.886849, +0.914404] |
| identity_delta_iou | 1.000000 | 1.000000 | +0.000000 | [+0.000000, +0.000000] |
| identity_delta_mse | 0.000000 | 0.000000 | +0.000000 | [+0.000000, +0.000000] |

Raster2Seq replay exact rate: 0.000000 over 502 available paired files.

Identity Delta-IoU/Delta-MSE are no-change diagnostics, not editing scores.
