# Structural Error Strata

| Error | Inside | Boundary 1 | Boundary 2 | Boundary 5 | Far outside |
|---|---:|---:|---:|---:|---:|
| node_false_positive | 560 | 10 | 12 | 30 | 341 |
| node_false_negative | 3116 | 53 | 46 | 141 | 1604 |
| edge_false_positive | 615 | 20 | 28 | 51 | 633 |
| edge_false_negative | 6029 | 216 | 211 | 606 | 4773 |
| node_deletion | 22 | 1 | 1 | 5 | 43 |
| node_insertion | 2451 | 35 | 24 | 117 | 1452 |
| node_substitution | 486 | 8 | 10 | 37 | 354 |
| edge_deletion | 435 | 22 | 28 | 58 | 626 |
| edge_insertion | 4912 | 155 | 155 | 451 | 3795 |

| Metric / covariate | Spearman rho | p-value |
|---|---:|---:|
| primary_node_f1 / target_node_count | -0.273897 | 6.61373e-44 |
| primary_node_f1 / target_edge_count | -0.225386 | 6.4118e-30 |
| primary_node_f1 / target_opening_count | NA | NA |
| primary_node_f1 / complexity_score | -0.242358 | 1.81311e-34 |
| primary_edge_overlap / target_node_count | -0.296558 | 1.64463e-51 |
| primary_edge_overlap / target_edge_count | -0.266650 | 1.27345e-41 |
| primary_edge_overlap / target_opening_count | NA | NA |
| primary_edge_overlap / complexity_score | -0.277564 | 4.34056e-45 |
| exact_or_fallback_ged / target_node_count | 0.628030 | 4.2183e-272 |
| exact_or_fallback_ged / target_edge_count | 0.605273 | 1.02057e-247 |
| exact_or_fallback_ged / target_opening_count | NA | NA |
| exact_or_fallback_ged / complexity_score | 0.616416 | 2.08561e-259 |

Node-loss top-10% share: 0.342778.
GED top-10% share: 0.316840.
