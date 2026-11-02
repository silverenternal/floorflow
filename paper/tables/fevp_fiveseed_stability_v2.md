# FEVP Five-Seed Stability

Status: stability evidence for the aggregate FEVP row set. The current
per-operation submission table is `residential_sota_main_v5.md`.

| Metric | Mean | SD | 95% CI | Best seed | Worst seed |
|---|---:|---:|---:|---:|---:|
| delta_iou | 0.987084 | 0.002096 | [0.984482, 0.989686] | 23 (0.990325) | 22 (0.984774) |
| delta_mse | 0.000218 | 0.000037 | [0.000171, 0.000264] | 19 (0.000168) | 20 (0.000259) |
| outside_pixel_exact | 1.000000 | 0.000000 | [1.000000, 1.000000] | 19 (1.000000) | 23 (1.000000) |
| outside_room_level_exact | 1.000000 | 0.000000 | [1.000000, 1.000000] | 19 (1.000000) | 23 (1.000000) |
| outside_edge_exact | 1.000000 | 0.000000 | [1.000000, 1.000000] | 19 (1.000000) | 23 (1.000000) |
| locality_violation_rate | 0.000000 | 0.000000 | [0.000000, 0.000000] | 19 (0.000000) | 23 (0.000000) |
| validity | 0.979750 | 0.001438 | [0.977965, 0.981535] | 19 (0.981848) | 23 (0.977814) |
| conditional_watertightness | 0.962637 | 0.003839 | [0.957871, 0.967404] | 23 (0.967491) | 19 (0.957875) |
| node_precision | 0.999361 | 0.000102 | [0.999234, 0.999488] | 22 (0.999505) | 23 (0.999253) |
| node_recall | 0.969127 | 0.002072 | [0.966554, 0.971700] | 19 (0.970104) | 21 (0.965421) |
| node_f1 | 0.982250 | 0.001163 | [0.980806, 0.983694] | 19 (0.982832) | 21 (0.980172) |
| edge_overlap | 0.929475 | 0.003978 | [0.924536, 0.934414] | 20 (0.932463) | 21 (0.922764) |
| ged | 0.869544 | 0.065263 | [0.788510, 0.950579] | 20 (0.824526) | 21 (0.983461) |
| typed_edge_f1 | 0.890007 | 0.001907 | [0.887640, 0.892375] | 20 (0.891356) | 21 (0.886686) |
| typed_ged | 2.120855 | 0.056962 | [2.050128, 2.191583] | 20 (2.076240) | 21 (2.218637) |

Five seeds are deterministic stability samples. Confidence intervals describe seed variation and are not population-sampling inference.

CI method: Student-t interval over five seed means; not case-level bootstrap.
