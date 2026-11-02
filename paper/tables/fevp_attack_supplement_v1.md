# FEVP Attack Coverage Supplement

| Metric | Submetric | Attack | Baseline | Attacked | Sensitive |
|---|---|---|---:|---:|---|
| delta_iou | — | remove_one_expected_change_pixel | 1.000000 | 0.984375 | True |
| delta_mse | — | remove_one_expected_change_pixel | 0.000000 | 0.003906 | True |
| conditional_watertightness | hole_area_ratio | enlarge_hole_area | 0.000000 | 0.008081 | True |
| outside_exact | outside_room_polygon_exact | move_room_polygon_one_pixel | 1.000000 | 0.000000 | True |
| outside_exact | outside_pixel_exact | move_room_pixels_outside_edit_region | 1.000000 | 0.866259 | True |
| ged | ged | relabel_one_graph_node | 0.000000 | 1.000000 | True |

Covered metric/submetric pairs: 27.
Required metric/submetric pairs covered: True.
