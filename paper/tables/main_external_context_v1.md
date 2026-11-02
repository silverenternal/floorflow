# Main External Context v1

Status: frozen context table. Rows use different datasets, inputs, and task
protocols. No row forms a cross-dataset leaderboard.

| Source | Task / input | Delta-IoU | Delta-MSE | Node F1 | Edge | GED | Allowed interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| FloorFlow oracle control | CubiCasa test503; target-derived oracle controls | 0.956703-0.999801 | 0.000019-0.000096 | N/A | N/A | N/A | Scoped editing anchor; no leaderboard rank |
| FloorFlow typed diagnostic | 12,395 strict-locality rows | 0.937473 | N/A | 0.955068 | 0.890007 | 2.120855 | Composition-policy diagnostic |
| FloorFlow predicted fidelity | CubiCasa val491, target-free controls | N/A | N/A | 0.660202 | 0.264622 | 11.600815 | Reference fidelity remains weak |
| HouseMind-E | RPLAN natural-language editing, published | 0.608000 | 0.001900 | 0.998000 | 0.934000 | 0.467000 | Published cross-dataset context only |
| Raster2Seq local redraw | CubiCasa edit rows; outside-frozen wrapper | 0.137160-0.189491 | N/A | N/A | N/A | N/A | Reconstruction wrapper, not native editing |
| Raster2Seq official | CubiCasa reconstruction, RoomF1 | N/A | N/A | N/A | N/A | N/A | RoomF1 `0.885268` public anchor |
| HouseMind official code | Public assets | N/A | N/A | N/A | N/A | N/A | Serialization unavailable; no reproduction claim |

FloorFlow oracle Delta-IoU and Delta-MSE numerically exceed the published
HouseMind-E pair, but the datasets and input contracts differ. FloorFlow
typed graph metrics remain below HouseMind-E graph metrics and predicted
fidelity remains weak. No complete HouseMind defeat, RPLAN leaderboard rank,
or full-CAD SOTA is claimed. Predicted room macro IoU remains 0.269701 and
is reported as a limitation, not a hidden zero.

<!-- registry bindings: external.raster2seq.official_room_f1, external.housemind_e.delta_iou, external.housemind_e.delta_mse, external.housemind_e.node_f1, external.housemind_e.edge_overlap, external.housemind_e.ged, predicted.fidelity.room_macro_iou, predicted.fidelity.node_f1, predicted.fidelity.edge_overlap, predicted.fidelity.ged, external.housemind.status -->
