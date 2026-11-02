# Aggressive CVPR Target Audit

| Requested target | Status | Evidence |
|---|---|---|
| oracle_control_edit_delta_anchor | passed | CubiCasa test503, five operations and seeds: Delta-IoU 0.956703-0.999801 and Delta-MSE 0.000019-0.000096 exceed published HouseMind-E anchors 0.608/0.0019 under aligned metric definitions. Different datasets and inputs; not a leaderboard rank. |
| oracle_control_strict_locality | passed | Exact outside pixel/room/polygon/wall/opening/edge preservation 1.0, FP-Eval validity 1.0, conditional watertightness 0.9579-0.9675. Oracle controls are non-deployable. |
| beat_house_mind_e_complete_method | blocked | HouseMind-E complete-method superiority remains blocked. Use its published Table 3 values only as non-paired benchmark context. |
| rplan_sota | blocked | Do not claim RPLAN or complete-CAD SOTA. The current complete-CAD evidence is a target-free CubiCasa diagnostic with substantially weaker geometric metrics than the oracle-control identity task. |
| floorset_sota | blocked | FloorSet is chip-floorplan work and is excluded by the user's no-chips instruction. |
| complete_cad_sota | blocked | No matched RPLAN/full-CAD data or executable baseline; current target-free CAD diagnostics are substantially weaker. |
| free_text_editing_sota | failed | Held-out deterministic free-text editor reaches Delta-IoU 0.527092 and Delta-MSE 0.012257, below the published HouseMind-E anchors. |
| all_graph_protocol_sota | blocked | Aligned FloorFlow reaches Node F1 0.982250, edge overlap 0.929475, and GED 0.869544, below HouseMind-E 0.998/0.934/0.467. Canonical and typed graphs have no matched external leaderboard. |
| paired_room_iou_improvement | passed | On CubiCasa5K test509, validation-frozen polygon fusion preserves RoomF1 0.898948, raises Room IoU to 0.625051 with paired 95% CI [0.000779, 0.002794], and leaves RoomF1 exactly unchanged for all 509 cases. |
