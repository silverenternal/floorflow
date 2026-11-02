# Structural Error Strata

| Error | Inside | Boundary 1 | Boundary 2 | Boundary 5 | Far outside |
|---|---:|---:|---:|---:|---:|
| node_false_positive | 4009 | 242 | 251 | 746 | 15022 |
| node_false_negative | 1342 | 74 | 70 | 229 | 3524 |
| edge_false_positive | 21 | 0 | 3 | 1 | 75 |
| edge_false_negative | 2890 | 176 | 193 | 499 | 9398 |
| node_deletion | 2852 | 172 | 176 | 501 | 11573 |
| node_insertion | 51 | 2 | 5 | 11 | 174 |
| node_substitution | 1074 | 52 | 54 | 154 | 2876 |
| edge_deletion | 5687 | 296 | 323 | 1013 | 19728 |
| edge_insertion | 656 | 42 | 42 | 108 | 2789 |

| Metric / covariate | Spearman rho | p-value |
|---|---:|---:|
| primary_node_f1 / target_node_count | 0.490503 | 3.02841e-150 |
| primary_node_f1 / target_edge_count | 0.496551 | 1.70154e-154 |
| primary_node_f1 / target_opening_count | 0.234050 | 3.3915e-32 |
| primary_node_f1 / complexity_score | 0.405044 | 1.62219e-98 |
| primary_edge_overlap / target_node_count | 0.051984 | 0.00963331 |
| primary_edge_overlap / target_edge_count | 0.052031 | 0.00956908 |
| primary_edge_overlap / target_opening_count | -0.036605 | 0.0684158 |
| primary_edge_overlap / complexity_score | 0.017794 | 0.375849 |
| exact_or_fallback_ged / target_node_count | 0.031833 | 0.113069 |
| exact_or_fallback_ged / target_edge_count | 0.006082 | 0.762153 |
| exact_or_fallback_ged / target_opening_count | 0.114097 | 1.22218e-08 |
| exact_or_fallback_ged / complexity_score | 0.061346 | 0.00224518 |

Node-loss top-10% share: 0.157142.
GED top-10% share: 0.192061.
