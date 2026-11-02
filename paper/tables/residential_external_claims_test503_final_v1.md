# Matched Residential Reconstruction Evidence

| Metric | FloorFlow predicted completion | Raster2Seq matched | Delta | Paired 95% CI |
|---|---:|---:|---:|---:|
| room_macro_iou | 0.068437 | 0.027212 | +0.041225 | [+0.035828, +0.046711] |
| room_support_iou | 0.793501 | 0.472832 | +0.320669 | [+0.305161, +0.335659] |
| node_f1 | 0.537526 | 0.349905 | +0.187621 | [+0.167207, +0.208586] |
| edge_overlap | 0.139266 | 0.064303 | +0.074963 | [+0.060731, +0.088707] |
| ged | 11.898608 | 13.423459 | -1.524851 | [-2.073559, -0.976143] |
| wall_f1 | 0.298909 | 0.000000 | +0.298909 | [+0.287108, +0.310359] |
| opening_f1 | 0.313125 | 0.008927 | +0.304198 | [+0.291227, +0.316684] |
| furniture_f1 | 0.015905 | 0.015905 | +0.000000 | [+0.000000, +0.000000] |
| pixel_accuracy | 0.170248 | 0.053026 | +0.117223 | [+0.106801, +0.127680] |

| Claim branch | Enabled |
|---|---:|
| matched_cubicasa_reconstruction_head_to_head | True |
| complete_cad_generation_sota | False |
| rplan_generation_sota | False |
| house_mind_editing_defeat | False |
| strict_locality_protocol_contribution | True |

Raster2Seq is reconstruction-only. HouseMind-E, CE2EPlan, RPLAN, TLC-Plan, and FMLM remain context-only or unavailable.
