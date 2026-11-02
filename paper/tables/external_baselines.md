# External baseline table

## CubiCasa oracle-program

| Method | Cases | Room support IoU | Footprint coverage | Outside pixels | Condition |
|---|---:|---:|---:|---:|---|
| House-GAN++ | 64 | 0.13366 | 0.18626 | 3254.38 | GT room program |
| HouseDiffusion | 58 converted / forward validated | not sampled | — | — | GT room program |

Source: `outputs/reports/external_baseline_comparison_v1.json`.

## Native RPLAN boundary-only

| Method | Cases | Footprint coverage | Outside pixels | Condition |
|---|---:|---:|---:|---|
| MaskPLAN Large/Single/cross | 64 | 1.00000 | 614.72 | boundary only |

MaskPLAN is evaluated on its native RPLAN format. It is not pooled with the
CubiCasa main result. Oracle-program rows are target-derived controls and must
not be presented as contour-only baselines.
