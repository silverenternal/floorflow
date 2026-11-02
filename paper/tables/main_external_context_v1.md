# Main External Context v1

Status: context table. No row forms a cross-dataset leaderboard.

| Source | Task | Evidence | Allowed interpretation |
|---|---|---|---|
| HouseMind-E | RPLAN natural-language editing | Delta-IoU 0.608, Delta-MSE 0.0019, Node F1 0.998, edge 0.934, GED 0.467 | Published protocol anchor |
| HouseMind-G | RPLAN generation | Published generation results | Generation context only |
| Raster2Seq | CubiCasa reconstruction | RoomF1 0.887 public anchor | Matched reconstruction baseline |
| Raster2Seq local redraw | CubiCasa edit rows | Delta-IoU 0.035690-0.164930 | Outside-frozen reconstruction wrapper |
| CE2EPlan | RPLAN generation | Published and reproduced generation metrics | Generation-only context |

FloorFlow oracle-control Delta-IoU and Delta-MSE exceed the HouseMind-E anchors.
FloorFlow aligned Node F1, edge overlap, and GED remain below the published
HouseMind-E values.

No complete HouseMind method, RPLAN leaderboard, or full-CAD SOTA is claimed.
