# Complexity Strata

| Family | Bin | N | Delta-IoU | Delta-MSE | Outside exact | Validity | Watertight | NodeF1 | Edge | GED |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| room_count | 0-4 | 1495 | 0.977269 | 0.000187 | 1.000000 | 1.000000 | 0.930435 | 0.959899 | 0.935619 | 0.266221 |
| room_count | 5-7 | 4270 | 0.983964 | 0.000318 | 1.000000 | 1.000000 | 0.982201 | 0.924094 | 0.888008 | 1.614520 |
| room_count | 8+ | 6630 | 0.991307 | 0.000160 | 1.000000 | 1.000000 | 0.965611 | 0.933024 | 0.880785 | 2.288386 |
| edit_area_fraction | small | 4130 | 0.981438 | 0.000045 | 1.000000 | 1.000000 | 0.968765 | 0.980542 | 0.970779 | 0.761501 |
| edit_area_fraction | medium | 4130 | 0.985450 | 0.000095 | 1.000000 | 1.000000 | 0.966102 | 0.951071 | 0.917513 | 1.434625 |
| edit_area_fraction | large | 4135 | 0.994355 | 0.000513 | 1.000000 | 1.000000 | 0.966385 | 0.868032 | 0.781500 | 3.239178 |
| interface_complexity | both | 7455 | 0.985930 | 0.000140 | 1.000000 | 1.000000 | 0.970355 | 0.942748 | 0.900931 | 1.753186 |
| interface_complexity | wall_only | 4395 | 0.989775 | 0.000290 | 1.000000 | 1.000000 | 0.970193 | 0.927363 | 0.879264 | 1.874175 |
| interface_complexity | opening_only | 300 | 0.968134 | 0.000194 | 1.000000 | 1.000000 | 0.930000 | 0.960786 | 0.941236 | 1.040000 |
| interface_complexity | neither | 245 | 0.997134 | 0.001298 | 1.000000 | 1.000000 | 0.857143 | 0.713043 | 0.681531 | 3.448980 |

Edit-area cut points: [0.07527119769758689, 0.16180311116157603].
All rows are CubiCasa test503 oracle-control rows. Uncertainty uses floorplan-cluster bootstrap and Holm correction for confirmatory Delta-IoU, Delta-MSE, Node F1, and GED deviations.
