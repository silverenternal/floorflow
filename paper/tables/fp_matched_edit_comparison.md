# Matched Edit-Detection Comparison

| Operation | Cases | FloorFlow Delta-IoU | Raster2Seq Delta-IoU | Raster2Seq Delta-MSE | Raster2Seq outside exact | FloorFlow outside exact |
|---|---:|---:|---:|---:|---:|---:|
| room_type | 487 | 0.8353 (0.8108-0.8528) | 0.1249 | 0.1579 | 0.9545 | 1.0000 |
| room_add | 495 | 0.8372 (0.8263-0.8458) | 0.0991 | 0.1739 | 0.8773 | 1.0000 |
| room_remove | 500 | 0.8352 (0.7964-0.8676) | 0.1649 | 0.1518 | 0.9573 | 1.0000 |
| opening_delete | 502 | 0.9616 (0.9539-0.9713) | 0.0357 | 0.0882 | 0.9178 | 1.0000 |
| furniture_delete (supp.) | 495 | 0.9734 (0.9693-0.9761) | 0.0572 | 0.1188 | 0.8969 | 1.0000 |

Raster2Seq independently regenerates both sides from the same coarse-control line renderings. It is not edit-aware.
