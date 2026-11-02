# Metric hierarchy and link budget

| Level | Metric | Value |
| --- | --- | ---: |
| L0_legal | fpeval_validity | 1.00000 |
| L0_legal | room_target_recall | 0.99602 |
| L0_legal | outside_footprint_pixels | 0.00000 |
| L0_legal | conditional_watertight | 0.98874 |
| L1_topology | fpeval_watertight | 0.92445 |
| L1_topology | opening_wall_alignment | 0.46073 |
| L1_topology | furniture_room_containment | 0.84095 |
| L1_topology | room_component_gap | 3.49503 |
| L2_objects | wall_iou | 0.30033 |
| L2_objects | opening_iou | 0.18201 |
| L2_objects | opening_f1 | 0.30518 |
| L2_objects | furniture_iou | 0.06856 |
| L2_objects | furniture_f1 | 0.12488 |
| L3_instances | instance_geometric_iou | 0.51673 |
| L3_instances | instance_room_f1 | 0.18142 |
| L3_instances | instance_type_accuracy | 0.42750 |
| L3_instances | room_group_macro_iou | 0.20038 |
| L4_semantics | room_macro_iou | 0.16752 |
| L4_semantics | room_dataset_macro_iou | 0.08675 |
| L4_semantics | room_dataset_micro_iou | 0.19511 |
| L4_semantics | boundary_f1 | 0.27360 |
| L5_generation | style_separation | 0.04523 |
| L5_generation | seed_iou_gap | 0.00000 |
| L5_generation | diversity | 7.67087 |
| L5_generation | runtime_seconds | 0.30998 |

| Link | Metric | Gap | Owner |
| --- | --- | ---: | --- |
| style | worst_style_room_iou | 0.88393 | S2 style-conditioned room graph |
| room_segmenter | direct_segmenter_room_iou | 0.88862 | S2 dynamic room hierarchy |
| recursion | recursion_room_gain | 0.02554 | S2 recursive partition graph |
| coarse_layout | room_macro_iou | 0.83248 | S2 coarse-to-fine room tokens |
| objects | furniture_iou | 0.93144 | S3 dynamic object queries |
| composition | furniture_room_containment | 0.15905 | S3 layer-aware loss and legalization |
| topology | unconditional_watertight | 0.07555 | S3 dynamic wall topology |
| vectorization | fpeval_room_polygon_f1 | 0.82183 | S5 vector-readout audit |
| identifiability | strict_0_95_identifiable_fraction | 1.00000 | diagnostic ceiling only |

Dominant failing link: **identifiability**.
Oracle and identifiability ceilings are kept separate from predicted-chain metrics.
