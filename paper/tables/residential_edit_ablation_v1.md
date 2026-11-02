# Residential Edit Ablation

| Operation | Raster2Seq local | No typed repair | Raw FloorFlow | Typed completion | Gain |
|---|---:|---:|---:|---:|---:|
| room_type | 0.124879 | 0.904123 | 0.998975 | 0.999199 | +0.095076 |
| room_add | 0.099114 | 0.993935 | 0.999579 | 0.999801 | +0.005865 |
| room_remove | 0.164930 | 0.942246 | 0.996232 | 0.999756 | +0.057510 |
| opening_delete | 0.035690 | 0.437431 | 0.956703 | 0.782136 | +0.344705 |
| furniture_delete | 0.057240 | 0.890304 | 0.987812 | 0.891802 | +0.001499 |
