# Residential Main Results

Status: historical version. Current metrics: `residential_sota_main_v5.md`.

## Reconstruction

| Method | RoomF1 | Room IoU | Corner F1 | Semantic F1 | Opening F1 |
|---|---:|---:|---:|---:|---:|
| Selected interpolation | 0.887290 | 0.619114 | 0.596038 | 0.642397 | 0.357844 |

Published RoomF1 anchor: 0.887.

## Strict-Locality Editing

| Operation | Delta-IoU mean | Range | Delta-MSE | Pixel/room/polygon/edge exact | FP-Eval validity | FEVP five-check validity | Watertight | Matched R2S Delta-IoU | Paired p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| room_type | 0.999199 | 0.999199-0.999199 | 0.000069 | 1.0/1.0/1.0/1.0 | 1.0 | 0.9569 | 0.9579-0.9675 | 0.124879 | 6.02e-75 |
| room_add | 0.999801 | 0.999801-0.999801 | 0.000030 | 1.0/1.0/1.0/1.0 | 1.0 | 0.9556 | 0.9579-0.9675 | 0.099114 | 9.32e-81 |
| room_remove | 0.999756 | 0.999756-0.999756 | 0.000019 | 1.0/1.0/1.0/1.0 | 1.0 | 0.9540 | 0.9579-0.9675 | 0.164930 | 1.6e-74 |
| opening_delete | 0.956703 | 0.956703-0.956703 | 0.000096 | 1.0/1.0/1.0/1.0 | 1.0 | 0.9124 | 0.9579-0.9675 | 0.035690 | 4.31e-83 |
| furniture_delete | 0.987812 | 0.987812-0.987812 | 0.000083 | 1.0/1.0/1.0/1.0 | 1.0 | 0.9515 | 0.9579-0.9675 | 0.057240 | 3.04e-83 |

Control-gated FloorFlow preserves every audited outside layer exactly; typed before-repair is retained as a transferable baseline fix.

test509 reconstruction and test503 editing are separate tasks and are not pooled.
