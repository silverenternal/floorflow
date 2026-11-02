# Editing-Aware Raster2Seq Local Redraw

Status: earlier local-redraw analysis table. The current per-operation main
result and matched baselines are `residential_sota_main_v5.md`.

| Operation | Cases | FloorFlow Delta-IoU | Local-redraw Delta-IoU | FloorFlow Delta-MSE | Local-redraw Delta-MSE |
|---|---:|---:|---:|---:|---:|
| room_type | 487 | 0.9941 | 0.1372 | 0.000240 | 0.123740 |
| room_add | 495 | 0.9996 | 0.1668 | 0.000043 | 0.076352 |
| room_remove | 500 | 0.9941 | 0.1773 | 0.000606 | 0.122304 |
| opening_delete | 502 | 0.9635 | 0.1895 | 0.000082 | 0.014699 |
| furniture_delete | 495 | 0.9846 | 0.1454 | 0.000116 | 0.029982 |

The local-redraw baseline freezes outside pixels and permits Raster2Seq to redraw only inside the declared edit region. It is an editing-aware wrapper, not Raster2Seq's native task.
