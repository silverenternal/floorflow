# Matched Local Editing Baselines v1

Status: partial; oracle-control matched comparison is frozen, while the
predicted-control held-out comparison remains pending.

| Operation | Cases | FloorFlow IoU | R2S IoU | Editing-aware IoU | R2S p | Editing-aware Delta-IoU CI |
|---|---:|---:|---:|---:|---:|---:|
| room_type | 487 | 0.999199 | 0.124879 | 0.909800 | 6.019e-75 | [0.065883, 0.115083] |
| room_add | 495 | 0.999801 | 0.099114 | 0.995985 | 9.319e-81 | [0.000000, 0.010048] |
| room_remove | 500 | 0.999756 | 0.164930 | 0.998508 | 1.595e-74 | [0.000000, 0.004021] |
| opening_delete | 502 | 0.956703 | 0.035690 | 0.731151 | 4.311e-83 | [0.204670, 0.245953] |
| furniture_delete | 495 | 0.987812 | 0.057240 | 0.742469 | 3.044e-83 | [0.223465, 0.268625] |

Raster2Seq is a matched reconstruction wrapper, not a native
editing-aware method. The editing-aware row applies the same declared
after-edit controls through a local control-delta replacement.
HouseMind-E remains cross-dataset protocol context.
