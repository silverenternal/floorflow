# Residential CAGN five-arm validation

| Arm | Params | NDCG@5 | Selected legal | Selected violation | Invalid mass |
|---|---:|---:|---:|---:|---:|
| baseline | 0 | 0.7791 | 0.4738 | 0.5262 | 0.0 |
| hard_gate | 0 | 0.8193 | 0.5555 | 0.4445 | 0.0 |
| hard_plus_learnable | 4,487,829 | 0.9923 | 0.9965 | 0.0035 | 0.0 |
| full_constraint_embedding | 4,487,829 | 0.9962 | 0.9974 | 0.0026 | 0.0 |
| parameter_matched_random_gate | 4,487,829 | 0.7800 | 0.4685 | 0.5315 | 0.0 |
