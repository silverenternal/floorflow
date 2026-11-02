# Structural Error Strata

| Error | Inside | Boundary 1 | Boundary 2 | Boundary 5 | Far outside |
|---|---:|---:|---:|---:|---:|
| node_false_positive | 684 | 4 | 6 | 9 | 191 |
| node_false_negative | 1414 | 6 | 1 | 11 | 26 |
| edge_false_positive | 103 | 2 | 2 | 6 | 49 |
| edge_false_negative | 2720 | 56 | 64 | 150 | 393 |
| node_deletion | 110 | 4 | 4 | 5 | 185 |
| node_insertion | 805 | 6 | 0 | 6 | 55 |
| node_substitution | 245 | 3 | 1 | 3 | 15 |
| edge_deletion | 352 | 10 | 11 | 21 | 365 |
| edge_insertion | 1613 | 35 | 34 | 68 | 262 |

| Metric / covariate | Spearman rho | p-value |
|---|---:|---:|
| primary_node_f1 / target_node_count | -0.116796 | 5.46854e-09 |
| primary_node_f1 / target_edge_count | -0.088938 | 9.21843e-06 |
| primary_node_f1 / target_opening_count | NA | NA |
| primary_node_f1 / complexity_score | -0.098458 | 9.03054e-07 |
| primary_edge_overlap / target_node_count | -0.194947 | 1.17679e-22 |
| primary_edge_overlap / target_edge_count | -0.181666 | 7.76273e-20 |
| primary_edge_overlap / target_opening_count | NA | NA |
| primary_edge_overlap / complexity_score | -0.187064 | 5.87768e-21 |
| exact_or_fallback_ged / target_node_count | 0.212848 | 8.63176e-27 |
| exact_or_fallback_ged / target_edge_count | 0.195357 | 9.55471e-23 |
| exact_or_fallback_ged / target_opening_count | NA | NA |
| exact_or_fallback_ged / complexity_score | 0.202058 | 2.98302e-24 |

Node-loss top-10% share: 0.649913.
GED top-10% share: 0.644618.
