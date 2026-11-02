# Residential Main Results

Status: historical version. Current metrics: `residential_sota_main_v5.md`.

## Reconstruction

| Method | RoomF1 | Room IoU | Corner F1 | Opening F1 |
|---|---:|---:|---:|---:|
| Eight-view room-count consensus | 0.898948 | 0.623260 | 0.605224 | 0.360088 |

Published RoomF1 anchor: 0.887.

## Structural Consistency

| Protocol | Node F1 | Edge overlap | GED |
|---|---:|---:|---:|
| Typed matched routing | 0.955068 | 0.837563 | 2.120855 |
| HouseMind-style aligned | 0.982250 | 0.929475 | 0.869544 |
| Canonical direct-tiled | 0.933189 | 0.889887 | 1.812344 |

These graph protocols use different matching and readout rules and must not be compared as one leaderboard.

## Strict-Locality Editing

| Operation | Delta-IoU mean | Range | Delta-MSE | Pixel/room/polygon/edge exact | FP-Eval validity | FEVP five-check validity | Watertight | R2S baseline | Editing-aware baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| room_type | 0.999199 | 0.999199-0.999199 | 0.000069 | 1.0/1.0/1.0/1.0 | 1.0 | 0.9569 | 0.9579-0.9675 | 0.124879 | 0.909800 |
| room_add | 0.999801 | 0.999801-0.999801 | 0.000030 | 1.0/1.0/1.0/1.0 | 1.0 | 0.9556 | 0.9579-0.9675 | 0.099114 | 0.995985 |
| room_remove | 0.999756 | 0.999756-0.999756 | 0.000019 | 1.0/1.0/1.0/1.0 | 1.0 | 0.9540 | 0.9579-0.9675 | 0.164930 | 0.998508 |
| opening_delete | 0.956703 | 0.956703-0.956703 | 0.000096 | 1.0/1.0/1.0/1.0 | 1.0 | 0.9124 | 0.9579-0.9675 | 0.035690 | 0.731151 |
| furniture_delete | 0.987812 | 0.987812-0.987812 | 0.000083 | 1.0/1.0/1.0/1.0 | 1.0 | 0.9515 | 0.9579-0.9675 | 0.057240 | 0.742469 |

FloorFlow preserves every audited outside layer exactly and beats both the matched reconstruction wrapper and the editing-aware control baseline on every operation.

test509 reconstruction and test503 editing are separate tasks and are not pooled.
