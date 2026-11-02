# Residential Blocker Re-Review

| Item | Status |
|---|---|
| oracle_control_edit_delta_anchor | passed_scoped_metric_context |
| oracle_control_strict_locality | passed_oracle_only |
| house_mind_complete_method | blocked_after_partial_reproduction |
| rplan_sota | blocked |
| floorset_sota | excluded |
| free_text_editing_sota | failed |
| all_graph_protocol_sota | blocked |

HouseMind has a partial reproduction only: public weights load and emit spatial tokens, but the official image-to-token through post-processing chain remains unavailable.

The passed oracle entries are not deployable predictions and are not a HouseMind
RPLAN leaderboard result. The current positive evidence is the per-operation
Delta-IoU/Delta-MSE result and exact outside preservation.
