# FEVP v1.0 Protocol Coverage Matrix

A dash means not reported under FEVP or not comparable; it does not mean failure.

| Method | Outside exact | Validity | Watertight | Node F1 | Edge overlap | Delta-IoU |
|---|---:|---:|---:|---:|---:|---:|
| FloorFlow | 1.000/1.000 | 0.980 | 0.963 | 0.982 | 0.929 | 0.987 |
| HouseMind-E | — | — | — | 0.998 | 0.934 | 0.608 |
| HouseMind-O | — | — | — | 0.997 | 0.908 | 0.598 |
| HouseMind-G | — | — | — | — | — | — |
| Raster2Seq native | 0.921 px | — | — | — | — | 0.096 |
| Raster2Seq outside-frozen wrapper | 1.000 px | — | — | — | — | 0.163 |
| Raster2Seq official anchor | — | — | — | — | — | — |
| CE2EPlan | — | — | — | — | — | — |

FloorFlow is the only complete evaluated FEVP row. The matrix does not rank methods on missing values.
