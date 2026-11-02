# Matched Local Editing Baselines

Status: scoped oracle-control comparison complete. Predicted-control
fresh held-out comparison is optional and not evaluated.

| Operation | Cases | FloorFlow IoU | R2S IoU | Editing-aware IoU | R2S p | Editing-aware Delta-IoU CI |
|---|---:|---:|---:|---:|---:|---:|
| room_type | 487 | 0.999199 | 0.137160 | 0.909800 | 6.019e-75 | [0.065883, 0.115083] |
| room_add | 495 | 0.999801 | 0.166756 | 0.995985 | 9.319e-81 | [0.000000, 0.010048] |
| room_remove | 500 | 0.999756 | 0.177266 | 0.998508 | 1.595e-74 | [0.000000, 0.004021] |
| opening_delete | 502 | 0.956703 | 0.189491 | 0.731151 | 2.902e-82 | [0.204670, 0.245953] |
| furniture_delete | 495 | 0.987812 | 0.145367 | 0.742469 | 3.044e-83 | [0.223465, 0.268625] |

Raster2Seq is a matched reconstruction wrapper, not a native
editing-aware method; its exact outside preservation follows from the
outside freeze. The editing-aware row applies the declared after-edit
controls through an intentionally simple local control-delta paint.
Its FloorFlow comparator is recomputed by the baseline runner, so its
paired interval uses that run's own before-state rather than mixing
rows across consumers. HouseMind-E remains cross-dataset context.
