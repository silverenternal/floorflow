# Matched Residential Reconstruction Evidence

| Metric | FloorFlow predicted completion | Raster2Seq matched | Delta | Paired 95% CI |
|---|---:|---:|---:|---:|
| room_macro_iou | 0.074360 | 0.027811 | +0.046549 | [+0.040096, +0.053204] |
| room_support_iou | 0.797531 | 0.476247 | +0.321284 | [+0.305633, +0.336743] |
| node_f1 | 0.543851 | 0.345417 | +0.198433 | [+0.174716, +0.222522] |
| edge_overlap | 0.142191 | 0.075810 | +0.066381 | [+0.048682, +0.082975] |
| ged | 11.877800 | 13.234216 | -1.356415 | [-1.920570, -0.777902] |
| wall_f1 | 0.311268 | 0.000000 | +0.311268 | [+0.299349, +0.323145] |
| opening_f1 | 0.312944 | 0.008815 | +0.304129 | [+0.290470, +0.317210] |
| furniture_f1 | 0.020367 | 0.020367 | +0.000000 | [+0.000000, +0.000000] |
| pixel_accuracy | 0.177332 | 0.051561 | +0.125772 | [+0.114654, +0.136343] |

| Claim branch | Enabled |
|---|---:|
| matched_cubicasa_reconstruction_head_to_head | True |
| complete_cad_generation_sota | False |
| rplan_generation_sota | False |
| house_mind_editing_defeat | False |
| strict_locality_protocol_contribution | True |

Raster2Seq is reconstruction-only. HouseMind-E, CE2EPlan, RPLAN, TLC-Plan, and FMLM remain context-only or unavailable.
