# Delta Fidelity by Edit Strength

| Group | Cases | Delta-IoU | IoU 95% interval | IoU P10 | Delta-MSE | Outside exact | Violation |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 room | 10945 | 0.987674 | [0.986214, 0.989134] | 0.994472 | 0.000185808 | 1.000000 | 0.000000000 |
| 2-3 rooms | 1315 | 0.981359 | [0.977437, 0.985280] | 0.965065 | 0.000422298 | 1.000000 | 0.000000000 |
| 4+ rooms | 135 | 0.995016 | [0.993238, 0.996793] | 0.985422 | 0.000802395 | 1.000000 | 0.000000000 |
| room deletion | 2500 | 0.994120 | [0.992525, 0.995714] | 0.991206 | 0.000605869 | 1.000000 | 0.000000000 |
| opening/furniture edit | 4985 | 0.973938 | [0.970939, 0.976937] | 0.960674 | 0.000098800 | 1.000000 | 0.000000000 |

| Operation | Node F1 | Edge F1 | GED |
|---|---:|---:|---:|
| furniture_delete | 0.999724 | 0.993716 | 0.149091 |
| opening_delete | 0.999671 | 0.995020 | 0.118327 |
| room_add | 0.998241 | 0.937062 | 1.409697 |
| room_remove | 0.864368 | 0.717480 | 5.418000 |
| room_type | 0.912943 | 0.805654 | 3.526899 |

Delta fidelity is reported under strict locality. Room deletion is retained as the principal graph-distance tail rather than hidden inside the aggregate GED.

The intervals are descriptive case-level Student-t intervals, not five-seed population inference.
