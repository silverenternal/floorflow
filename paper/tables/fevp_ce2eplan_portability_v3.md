# CE2EPlan FEVP Portability Audit

| Family | Metric | Value |
|---|---|---:|
| Graph / parsing | parsable | 1.000000 |
| Graph / parsing | node_precision | 0.959750 |
| Graph / parsing | node_recall | 0.945583 |
| Graph / parsing | node_f1 | 0.947889 |
| Graph / parsing | edge_precision | 0.803071 |
| Graph / parsing | edge_recall | 0.769772 |
| Graph / parsing | edge_f1 | 0.770282 |
| Graph / parsing | edge_overlap | 0.646296 |
| Graph / parsing | ged | 2.975000 |
| Graph / parsing | ged_exact | 1.000000 |
| Validity | validity | 0.960000 |
| Validity | outer_boundary_closed | 1.000000 |
| Validity | room_polygon_closed | 1.000000 |
| Validity | no_self_intersection | 0.980000 |
| Validity | no_room_overlap | 0.980000 |
| Validity | semantic_complete | 1.000000 |
| Watertightness | watertight_rate | 1.000000 |
| Watertightness | hole_count | 0.000000 |
| Watertightness | hole_area_ratio | 0.000000 |
| Watertightness | disconnected_component_count | 0.000000 |
| Watertightness | dangling_endpoint_rate | 0.000000 |

Outside exact and violation rate: N/A because CE2EPlan is generation-only.

The FEVP evaluator is portable to a non-FloorFlow generation output: it decomposes CE2EPlan validity, watertightness, and graph quality on the same 200 public rows. This is not an editing or leaderboard comparison, and outside exact is not applicable.
