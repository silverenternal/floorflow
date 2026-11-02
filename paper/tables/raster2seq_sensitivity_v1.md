# Raster2Seq Matched-Preprocessing Sensitivity

| Regime | RoomF1 | Room IoU | Corner F1 |
|---|---:|---:|---:|
| checkpoint alpha 0.00 | 0.874352 | 0.613337 | 0.584998 |
| checkpoint alpha 0.25 | 0.879913 | 0.623329 | 0.591263 |
| checkpoint alpha 0.50 | 0.883728 | 0.633117 | 0.594406 |
| checkpoint alpha 0.75 | 0.884164 | 0.629625 | 0.595436 |
| checkpoint alpha 1.00 | 0.882119 | 0.632278 | 0.593677 |

View-only ranges on val503:

- RoomF1 0.885603-0.892615
- Room IoU 0.606739-0.636017
- Corner F1 0.599973-0.605695

| Edit operation | R2S Delta-IoU | FloorFlow Delta-IoU | R2S outside exact | FloorFlow outside exact | Paired p |
|---|---:|---:|---:|---:|---:|
| room_type | 0.124879 | 0.999199 | 0.954512 | 1.000000 | 6.019e-75 |
| room_add | 0.099114 | 0.999801 | 0.877315 | 1.000000 | 9.319e-81 |
| room_remove | 0.164930 | 0.999756 | 0.957327 | 1.000000 | 1.595e-74 |
| opening_delete | 0.035690 | 0.956703 | 0.917770 | 1.000000 | 4.311e-83 |
| furniture_delete | 0.057240 | 0.987812 | 0.896898 | 1.000000 | 3.044e-83 |
