# Residential Main Results

Status date: 2026-11-02. This is the canonical current residential result table.
It separates test509 identity, CubiCasa test503 oracle-control editing, and
protocol-aligned structural metrics.

## Reconstruction

| Method | RoomF1 | Room IoU | Corner F1 | Opening F1 |
|---|---:|---:|---:|---:|
| Eight-view consensus + polygon fusion | 0.898948 | 0.625051 | 0.605739 | 0.360088 |

Published RoomF1 anchor: 0.887.

## Structural Consistency

| Protocol | Node F1 | Edge overlap | GED |
|---|---:|---:|---:|
| Typed matched routing | 0.955068 | 0.837563 | 2.120855 |
| HouseMind-style aligned | 0.982250 | 0.929475 | 0.869544 |
| Canonical direct-tiled | 0.933189 | 0.889887 | 1.812344 |

These graph protocols use different matching and readout rules and must not be compared as one leaderboard.

## External Baseline Boundaries

| Baseline | Published / local evidence | Role |
|---|---|---|
| Raster2Seq | RoomF1 0.887 public anchor; local matched sensitivity is reported separately | matched reconstruction baseline; not editing-aware |
| CE2EPlan (TVCG 2026) | FID 1.08 published / 1.1270 reproduced; GED 3.48 / 5.2323 | generation-only RPLAN context |
| HouseMind-E | Delta-IoU 0.608, Delta-MSE 0.0019, Node F1 0.998, Edge 0.934, GED 0.467 | published reference anchor; full inference not reproduced |

HouseMind public weights were exercised on 16 prompts for 1,560 room tokens, but no image output was produced. These traces are not assigned benchmark metrics or direct-comparison status.

## Strict-Locality Editing

| Operation | Delta-IoU mean | Range | Delta-MSE | Pixel/room/polygon/edge exact | FP-Eval validity | FEVP five-check validity | Watertight | R2S baseline | Editing-aware baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| room_type | 0.999199 | 0.999199-0.999199 | 0.000069 | 1.0/1.0/1.0/1.0 | 1.0 | 0.9569 | 0.9579-0.9675 | 0.124879 | 0.909800 |
| room_add | 0.999801 | 0.999801-0.999801 | 0.000030 | 1.0/1.0/1.0/1.0 | 1.0 | 0.9556 | 0.9579-0.9675 | 0.099114 | 0.995985 |
| room_remove | 0.999756 | 0.999756-0.999756 | 0.000019 | 1.0/1.0/1.0/1.0 | 1.0 | 0.9540 | 0.9579-0.9675 | 0.164930 | 0.998508 |
| opening_delete | 0.956703 | 0.956703-0.956703 | 0.000096 | 1.0/1.0/1.0/1.0 | 1.0 | 0.9124 | 0.9579-0.9675 | 0.035690 | 0.731151 |
| furniture_delete | 0.987812 | 0.987812-0.987812 | 0.000083 | 1.0/1.0/1.0/1.0 | 1.0 | 0.9515 | 0.9579-0.9675 | 0.057240 | 0.742469 |

FloorFlow preserves every audited outside layer exactly and beats both the matched reconstruction wrapper and the editing-aware control baseline on every operation.

HouseMind-E publishes Delta-IoU `0.608` and Delta-MSE `0.0019`. FloorFlow
numerically exceeds both anchors for every operation under aligned definitions,
but the datasets and input contracts differ. The aligned Node F1, edge overlap,
and GED values do not exceed HouseMind-E and remain a limitation.

test509 reconstruction and test503 editing are separate tasks and are not pooled.
