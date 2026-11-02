# Edit-Type Strata v2

| Operation | Cases/seed | Delta-IoU | Delta-MSE | Outside exact | Watertight | Node F1 | Edge | GED |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| room_type | 487 | 0.994084 (0.9913-0.9990) | 0.000240207 | 1.000000 | 0.960280 | 0.989241 | 0.946461 | 0.557700 |
| room_add | 495 | 0.999568 (0.9989-0.9998) | 0.000042511 | 1.000000 | 0.962538 | 0.997664 | 0.978496 | 0.298182 |
| room_remove | 500 | 0.994120 (0.9911-0.9962) | 0.000605869 | 1.000000 | 0.962207 | 0.925392 | 0.743657 | 3.206400 |
| opening_delete | 502 | 0.963465 (0.9567-0.9730) | 0.000082042 | 1.000000 | 0.958616 | 0.999671 | 0.991176 | 0.118327 |
| furniture_delete | 495 | 0.984560 (0.9795-0.9886) | 0.000115795 | 1.000000 | 0.958716 | 0.999724 | 0.988863 | 0.149091 |

## Requested Operation Coverage

| Requested type | Status | Matched evidence |
|---|---|---|
| add_or_remove_room | available | room_add, room_remove |
| move_wall | unavailable | none |
| change_room_function | available | room_type |
| expand_or_shrink_room | unavailable | none |
| merge_or_split_room | partially_available | room_remove, room_add |
| multi_step_combined_edit | unavailable | none |

## GED Diagnosis

`room_remove` has the highest mean GED (3.206400). The decomposition separates node substitutions/deletions/insertions from edge deletions/insertions; edge edits dominate the tail.

No numeric claim is made for wall movement, room resizing, or sequential multi-step editing because matched rows are unavailable.
