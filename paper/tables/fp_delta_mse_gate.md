# Delta-MSE Improvement Gate

Status: historical aggregate. Current per-operation results:
`residential_sota_main_v5.md`.

| Variant | Delta-IoU | Delta-MSE | Outside exact | Conditional watertight |
|---|---:|---:|---:|---:|
| current local-delta | 0.8888 | 0.005254 | 1.0000 | 0.9583 |
| control-gated candidate | 0.9871 | 0.000218 | 1.0000 | 0.9579-0.9675 |

Decision: promote under the CubiCasa protocol; all five seeds pass the 0.95 topology gate.
