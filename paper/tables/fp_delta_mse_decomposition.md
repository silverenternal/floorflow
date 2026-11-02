# FloorFlow Delta-MSE Decomposition

Status: historical ungated decomposition. The current control-gated
per-operation result is `residential_sota_main_v5.md`.

| Region | Area fraction | Delta-MSE mean | Range | Error mass |
|---|---:|---:|---:|---:|
| global | 1.0000 | 0.005254 | 0.004660-0.006699 | 0.7359 |
| edit_region | 0.1551 | 0.024613 | 0.022962-0.029230 | 0.7359 |
| structure_boundary | 0.0411 | 0.039270 | 0.036164-0.048185 | 0.4059 |
| room_boundary | 0.0274 | 0.067724 | 0.064121-0.076327 | 0.3372 |
| edit_interior | 0.0882 | 0.002032 | 0.001134-0.003809 | 0.0132 |
| outside | 0.8511 | 0.000000 | 0.000000-0.000000 | 0.0000 |

HouseMind-E Delta-MSE anchor: 0.0019. The edit interior is below this anchor in 3/5 seeds.
