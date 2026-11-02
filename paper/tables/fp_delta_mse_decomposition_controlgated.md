# FloorFlow Control-Gated Delta-MSE Decomposition

Status: historical aggregate decomposition. Current per-operation headline:
`residential_sota_main_v5.md`.

| Region | Area fraction | Delta-MSE mean | Range | Error mass |
|---|---:|---:|---:|---:|
| global | 1.0000 | 0.000218 | 0.000168-0.000259 | 0.1700 |
| edit_region | 0.1551 | 0.001058 | 0.000820-0.001227 | 0.1700 |
| structure_boundary | 0.0411 | 0.001673 | 0.001308-0.001958 | 0.0820 |
| room_boundary | 0.0274 | 0.003024 | 0.001948-0.003729 | 0.0904 |
| edit_interior | 0.0882 | 0.000034 | 0.000028-0.000041 | 0.0030 |
| outside | 0.8511 | 0.000000 | 0.000000-0.000000 | 0.0000 |

HouseMind-E Delta-MSE anchor: 0.0019. The edit interior is below this anchor in 5/5 seeds.
