# Raster2Seq Outside-Edit Leakage

| Operation | Outside exact mean | p10 | min | Changed pixels mean | Changed pixels p90 | Leak fraction |
|---|---:|---:|---:|---:|---:|---:|
| room_add | 0.8773 | 0.6688 | 0.1281 | 1197.6 | 2754.0 | 12.2685% |
| furniture_delete (supp.) | 0.8969 | 0.7228 | 0.1943 | 1090.8 | 2476.6 | 10.3102% |
| opening_delete | 0.9178 | 0.7774 | 0.2454 | 980.1 | 2082.7 | 8.2230% |
| room_type | 0.9545 | 0.8675 | 0.3526 | 419.4 | 1027.4 | 4.5488% |
| room_remove | 0.9573 | 0.8883 | 0.0232 | 383.8 | 879.0 | 4.2673% |

Exact match is strict uint8 label equality outside the declared edit rectangle, excluding footprint exterior and dilated walls.
