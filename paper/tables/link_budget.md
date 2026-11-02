# Metric hierarchy and link budget

| Level | Metric | Value |
| --- | --- | ---: |
| L0_legal | fpeval_validity | 1.00000 |
| L0_legal | room_target_recall | 0.96578 |
| L0_legal | outside_footprint_pixels | 0.00000 |
| L0_legal | conditional_watertight | 0.98874 |
| L1_topology | fpeval_watertight | 0.92445 |
| L1_topology | opening_wall_alignment | 0.46065 |
| L1_topology | furniture_room_containment | 0.00000 |
| L1_topology | room_component_gap | 3.75149 |
| L2_objects | wall_iou | 0.30034 |
| L2_objects | opening_iou | 0.18202 |
| L2_objects | opening_f1 | 0.30519 |
| L2_objects | furniture_iou | 0.06855 |
| L2_objects | furniture_f1 | 0.12488 |
| L3_instances | instance_geometric_iou | 0.52363 |
| L3_instances | instance_room_f1 | 0.17621 |
| L3_instances | instance_type_accuracy | 0.43416 |
| L4_semantics | room_macro_iou | 0.17018 |
| L4_semantics | room_dataset_macro_iou | 0.08486 |
| L4_semantics | room_dataset_micro_iou | 0.19756 |
| L4_semantics | boundary_f1 | 0.27360 |
| L5_generation | style_separation | 0.05304 |
| L5_generation | seed_iou_gap | 0.00000 |
| L5_generation | runtime_seconds | 2.20889 |

| Link | Metric | Gap | Owner |
| --- | --- | ---: | --- |
| style | worst_style_room_iou | 0.88770 | S2 style-conditioned room graph |
| room_segmenter | direct_segmenter_room_iou | 0.83844 | S2 dynamic room hierarchy |
| recursion | recursion_room_gain | 0.03258 | S2 recursive partition graph |
| coarse_layout | room_macro_iou | 0.82982 | S2 coarse-to-fine room tokens |
| objects | furniture_iou | 0.93145 | S3 dynamic object queries |
| composition | furniture_room_containment | 1.00000 | S3 layer-aware loss and legalization |
| topology | unconditional_watertight | 0.07555 | S3 dynamic wall topology |
| vectorization | fpeval_room_polygon_f1 | 0.82654 | S5 vector-readout audit |
| identifiability | strict_0_95_identifiable_fraction | 1.00000 | diagnostic ceiling only |

Dominant failing link: **composition**.
Oracle and identifiability ceilings are kept separate from predicted-chain metrics.
