# Room Page-Floor Five-Seed Confirmation

| Metric | Holdout delta mean | Range | Improved seeds | CI-excluding seeds |
|---|---:|---:|---:|---:|
| room_macro_iou | +0.095561 | [+0.086502, +0.110607] | 5/5 | 5/5 |
| instance_room_f1 | +0.143787 | [+0.128665, +0.153932] | 5/5 | 5/5 |
| instance_count_abs_error | -3.203738 | [-3.457944, -3.018692] | 5/5 | 5/5 |
| instance_type_accuracy | +0.129393 | [+0.119626, +0.138941] | 5/5 | 5/5 |
| room_support_iou | +0.000000 | [+0.000000, +0.000000] | 0/5 | 0/5 |

Gate passed: `true`.

Case-clustered holdout bootstrap:

| Metric | Mean delta | 95% CI | Wilcoxon p | Rank-biserial |
|---|---:|---:|---:|---:|
| room_macro_iou | +0.095561 | [+0.073985, +0.116945] | 2.62963e-12 | +0.7865 |
| instance_room_f1 | +0.143787 | [+0.114442, +0.172838] | 1.24234e-13 | +0.8413 |
| instance_count_abs_error | -3.203738 | [-3.975701, -2.416822] | 3.54488e-11 | -0.7515 |
| instance_type_accuracy | +0.129393 | [+0.049392, +0.209239] | 0.00224271 | +0.4263 |
| room_support_iou | +0.000000 | [+0.000000, +0.000000] | 1 | +0.0000 |

