# FloorFlow Main Results v3

CubiCasa test503, declared oracle controls, control-gated local delta, five
repeated seeds. Current canonical source: `residential_sota_main_v5.md`.

## Editing quality

| Operation | Cases | Delta-IoU | Delta-MSE | Role |
|---|---:|---:|---:|---|
| room type | 487 | 0.999199 | 0.000069 | identity change |
| room add | 495 | 0.999801 | 0.000030 | partition insertion |
| room remove | 500 | 0.999756 | 0.000019 | merge |
| opening delete | 502 | 0.956703 | 0.000096 | object deletion |
| furniture delete | 495 | 0.987812 | 0.000083 | object deletion |

HouseMind-E publishes Delta-IoU `0.608` and Delta-MSE `0.0019`. FloorFlow
numerically exceeds both anchors for every operation under aligned definitions.
The datasets and input contracts differ, so this is not a same-benchmark rank.

## Strict locality

| Metric | Result | Definition |
|---|---:|---|
| Outside pixel/room/polygon/wall/opening/edge exact | 1.000000 | zero-tolerance preservation outside the declared edit region |

## Topology legality

| Metric | Result | Role |
|---|---:|---|
| Validity | 1.000000 | legal CAD structure |
| Independent FEVP five-check validity | 0.9124-0.9569 | stricter audit, reported separately |
| Conditional watertightness | 0.957875-0.967491 | closed topology on watertight targets |

## Graph consistency

| Metric | Result | Role |
|---|---:|---|
| Node F1 | 0.982250 (0.980172-0.982832) | room-type identity |
| Edge overlap | 0.929475 (0.922764-0.932463) | room-type adjacency |
| GED | 0.869544 (0.824526-0.983461) | graph consistency; below HouseMind-E 0.467 |
| Exact GED rate | 0.999355 | graph metric reliability |

GED variance is operation-driven: room removal dominates graph edits through node insertion/substitution, while opening and furniture deletion are near zero.

HouseMind-E references are cross-dataset protocol context only. Raster2Seq is a
matched reconstruction baseline, not a native editing method. Oracle controls
are target-derived and non-deployable.
