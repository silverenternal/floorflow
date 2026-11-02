# Strict-Locality Protocol Boundary

| Boundary | N | Delta-IoU | Delta-MSE | Outside exact | Room exact | Validity | Watertight | NodeF1 | GED |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low_room_count_0_4 | 1495 | 0.977269 | 0.000187 | 1.000000 | 1.000000 | 1.000000 | 0.930435 | 0.959899 | 0.266221 |
| weak_interface_opening_or_none | 545 | 0.981170 | 0.000690 | 1.000000 | 1.000000 | 1.000000 | 0.897248 | 0.849415 | 2.122936 |
| low_edit_signal_bottom_tercile | 827 | 0.967329 | 0.000071 | 1.000000 | 1.000000 | 1.000000 | 0.961306 | N/A | N/A |

| Arm | Wall breaks | Watertight | Outside exact |
|---|---:|---:|---:|
| control_off_guard_off | 79 | 0.963695 | 0.989987 |
| control_on_guard_off | 81 | 0.962888 | 1.000000 |
| control_off_guard_on | 77 | 0.964502 | 0.989987 |
| control_on_guard_on | 77 | 0.964502 | 1.000000 |

When edit signals are sparse or weak interfaces provide few topological constraints, conditional watertightness can fall below 0.95. Strict outside preservation remains intact under the control gate. This is a validity boundary of the current protocol, not a redefinition of the locality guarantee.
