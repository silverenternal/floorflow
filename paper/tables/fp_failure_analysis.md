# Failure Analysis

| Family | Multi-label cases | Fraction | Exclusive top-priority cases | Definition |
|---|---:|---:|---:|---|
| edit_boundary_wall_break | 60 | 0.004841 | 60 | conditional watertightness fails, junction F1 < 0.9, and the edit boundary intersects a wall |
| complex_home_degradation | 1498 | 0.120855 | 1490 | >=8 target rooms and Node F1 < 0.85 or GED > 5 |
| room_identity_ambiguity | 1744 | 0.140702 | 826 | Node F1 < 0.8 with a predicted merge or target split |
| wall_junction_mismatch | 1552 | 0.125212 | 1202 | target-matched junction F1 < 0.9 |

| Oracle-to-deployable stage | Watertight | Edit cost | Adjacency F1 |
|---|---:|---:|---:|
| Oracle clean | 0.969136 | 13.534188 | 0.766167 |
| Oracle 10% dropout | 0.882716 | 43.587628 | 0.474979 |
| Predicted control | 0.875000 | 221.173766 | N/A |

Failure thresholds are fixed in `build_fp_failure_analysis_v2.py`; multi-label families overlap, while exclusive counts use the listed priority order.
