# FloorFlow Main Results

Status: updated to the current per-operation evidence. The canonical table is
`residential_sota_main_v5.md`.

FloorFlow edit metrics use CubiCasa test503 with declared oracle controls,
control-gated local deltas, and five repeated seeds. HouseMind-E anchors come
from RPLAN natural-language editing and are shown as metric context only.

| Operation | Cases | Delta-IoU | Delta-MSE | Outside exact | FP-Eval validity |
|---|---:|---:|---:|---:|---:|
| room type | 487 | 0.999199 | 0.000069 | 1.000000 | 1.000000 |
| room add | 495 | 0.999801 | 0.000030 | 1.000000 | 1.000000 |
| room remove | 500 | 0.999756 | 0.000019 | 1.000000 | 1.000000 |
| opening delete | 502 | 0.956703 | 0.000096 | 1.000000 | 1.000000 |
| furniture delete | 495 | 0.987812 | 0.000083 | 1.000000 | 1.000000 |

HouseMind-E publishes Delta-IoU `0.608` and Delta-MSE `0.0019`. Every operation
numerically exceeds both anchors under aligned metric definitions, but the
datasets and input contracts differ. Conditional watertightness is
`0.9579-0.9675`; independent FEVP five-check validity is `0.9124-0.9569`.

## HouseMind-Style Structural Main Table

HouseMind-E reports RPLAN Node F1 0.998, Edge overlap 0.934, and GED 0.467. Those values are a protocol reference, not a same-dataset ranking.

| Metric | Mean | Range | HouseMind-E |
|---|---:|---:|---:|
| Node F1 | 0.982250 | 0.980172-0.982832 | 0.998 |
| Edge overlap | 0.929475 | 0.922764-0.932463 | 0.934 |
| GED | 0.869544 | 0.824526-0.983461 | 0.467 |
| Exact GED rate | 0.999355 | 0.999193-0.999597 | Not reported |

The aligned graph readout remains below the HouseMind-E public graph anchors.
This is a topology limitation, not an additional positive SOTA claim.

This structural table is HouseMind-protocol-aligned on CubiCasa oracle controls. It is not a HouseMind RPLAN leaderboard result.

## Canonical Instance Graph

| Metric | Mean | Range |
|---|---:|---:|
| Node F1 | 0.933189 | 0.928125-0.937616 |
| Edge overlap | 0.889887 | 0.877700-0.897818 |
| Unit GED | 1.812344 | 1.696248-1.968939 |
| Exact GED rate | 0.876886 | 0.874546-0.879387 |

The canonical instance graph is secondary because it applies one-to-one room-instance IoU matching, whereas the HouseMind-style view uses unique room-type sets and type-pair edges.
