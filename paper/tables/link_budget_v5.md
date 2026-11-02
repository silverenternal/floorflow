# Metric hierarchy and link budget

| Level | Metric | Value |
| --- | --- | ---: |
| L0_legal | fpeval_validity | 1.00000 |
| L0_legal | room_target_recall | 0.99602 |
| L0_legal | outside_footprint_pixels | 0.00000 |
| L0_legal | conditional_watertight | 0.98649 |
| L1_topology | fpeval_watertight | 0.92247 |
| L1_topology | wall_graph_f1 | 0.41955 |
| L1_topology | opening_wall_alignment | 0.45811 |
| L1_topology | furniture_room_containment | 0.81113 |
| L1_topology | room_component_gap | 4.50696 |
| L2_objects | wall_iou | 0.29244 |
| L2_objects | opening_iou | 0.18838 |
| L2_objects | opening_f1 | 0.31436 |
| L2_objects | furniture_iou | 0.06899 |
| L2_objects | furniture_f1 | 0.12597 |
| L3_instances | instance_geometric_iou | 0.51775 |
| L3_instances | instance_room_f1 | 0.15983 |
| L3_instances | instance_type_accuracy | 0.43194 |
| L3_instances | room_group_macro_iou | 0.20184 |
| L4_semantics | room_macro_iou | 0.16949 |
| L4_semantics | room_dataset_macro_iou | 0.08445 |
| L4_semantics | room_dataset_micro_iou | 0.19595 |
| L4_semantics | rare_class_recall | 0.88277 |
| L4_semantics | boundary_f1 | 0.27002 |
| L5_generation | style_separation | 0.03993 |
| L5_generation | seed_iou_gap | 0.00000 |
| L5_generation | diversity | 7.13300 |
| L5_generation | runtime_seconds | 0.34200 |

| Link | Metric | Gap | Owner |
| --- | --- | ---: | --- |
| style | worst_style_room_iou | 0.88512 | S2 style-conditioned room graph |
| room_segmenter | direct_segmenter_room_iou | 0.88995 | S2 dynamic room hierarchy |
| recursion | recursion_room_gain | 0.03195 | S2 recursive partition graph |
| coarse_layout | room_macro_iou | 0.83051 | S2 coarse-to-fine room tokens |
| objects | furniture_iou | 0.93101 | S3 dynamic object queries |
| composition | furniture_room_containment | 0.18887 | S3 layer-aware loss and legalization |
| topology | unconditional_watertight | 0.07753 | S3 dynamic wall topology |
| vectorization | fpeval_room_polygon_f1 | 0.83925 | S5 vector-readout audit |
| identifiability | strict_0_95_identifiable_fraction | 1.00000 | diagnostic ceiling only |

Dominant failing link: **identifiability**.
Oracle and identifiability ceilings are kept separate from predicted-chain metrics.
