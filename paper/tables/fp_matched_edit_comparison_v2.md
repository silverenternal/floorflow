# Shared-Project-Preprocessing Raster2Seq Sensitivity

Status: earlier paired sensitivity table. The current submission headline and
matched Raster2Seq range are in `residential_sota_main_v5.md`.

| Operation | Cases | FloorFlow Delta-IoU | Raster2Seq Delta-IoU | Raster2Seq Delta-MSE | Raster2Seq outside exact | FloorFlow outside exact |
|---|---:|---:|---:|---:|---:|---:|
| room_type | 487 | 0.9941 (0.9913-0.9990) | 0.1249 | 0.1579 | 0.9545 | 1.0000 |
| room_add | 495 | 0.9996 (0.9989-0.9998) | 0.0991 | 0.1739 | 0.8773 | 1.0000 |
| room_remove | 500 | 0.9941 (0.9911-0.9962) | 0.1649 | 0.1518 | 0.9573 | 1.0000 |
| opening_delete | 502 | 0.9635 (0.9567-0.9730) | 0.0357 | 0.0882 | 0.9178 | 1.0000 |
| furniture_delete (supp.) | 495 | 0.9846 (0.9795-0.9886) | 0.0572 | 0.1188 | 0.8969 | 1.0000 |

Both methods use shared project-controlled coarse-control renderings. This is not Raster2Seq's official preprocessing, and Raster2Seq is not edit-aware.
