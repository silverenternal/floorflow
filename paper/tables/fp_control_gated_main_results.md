# Control-Gated Local Delta Main Results

The current per-operation main results are:

| Operation | Cases | Delta-IoU | Delta-MSE | Outside exact | FP-Eval validity |
|---|---:|---:|---:|---:|---:|
| room type | 487 | 0.999199 | 0.000069 | 1.0000 | 1.0000 |
| room add | 495 | 0.999801 | 0.000030 | 1.0000 | 1.0000 |
| room remove | 500 | 0.999756 | 0.000019 | 1.0000 | 1.0000 |
| opening delete | 502 | 0.956703 | 0.000096 | 1.0000 | 1.0000 |
| furniture delete | 495 | 0.987812 | 0.000083 | 1.0000 | 1.0000 |

## Historical Aggregate Seed Table

| Seed | Delta-IoU | Delta-MSE | Outside pixel exact | Outside room exact | Validity | Conditional watertight |
|---:|---:|---:|---:|---:|---:|---:|
| 19 | 0.9878 | 0.000168 | 1.0000 | 1.0000 | 1.0000 | 0.9579 |
| 20 | 0.9863 | 0.000259 | 1.0000 | 1.0000 | 1.0000 | 0.9602 |
| 21 | 0.9862 | 0.000251 | 1.0000 | 1.0000 | 1.0000 | 0.9625 |
| 22 | 0.9848 | 0.000211 | 1.0000 | 1.0000 | 1.0000 | 0.9652 |
| 23 | 0.9903 | 0.000199 | 1.0000 | 1.0000 | 1.0000 | 0.9675 |

The seed table is retained as historical provenance. The per-operation table is
the current canonical evidence. Delta metrics use the HouseMind edit-detection
protocol on CubiCasa test503. This is a protocol anchor, not a HouseMind RPLAN
leaderboard result.
