# Matched Edit-Quality / Locality Pareto

Status: historical aggregate frontier. The current per-operation submission
result is `residential_sota_main_v5.md`.

| Method | Delta-IoU | Delta-MSE | Outside pixel exact | Outside leakage |
|---|---:|---:|---:|---:|
| FloorFlow control-gated | 0.987084 | 0.000217613 | 1.000000 | 0.000000 |
| FloorFlow raw local renderer | 0.853715 | 0.013384405 | 0.986115 | 0.013885 |
| Raster2Seq native | 0.096246 | 0.137954056 | 0.920721 | 0.079279 |
| Raster2Seq outside-frozen redraw | 0.163395 | 0.073185778 | 1.000000 | 0.000000 |

Delta-IoU/outside-exact frontier: FloorFlow control-gated.
Delta-MSE/leakage frontier: FloorFlow control-gated.

HouseMind-E RPLAN anchor: Delta-IoU 0.608, Delta-MSE 0.0019, outside exact not reported. It is excluded from the matched frontier.
