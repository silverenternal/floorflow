# Existing Raster Outputs Under the Canonical Room Graph

| Source | Node P | Node R | Node F1 | Edge overlap | Nodes | Edges |
|---|---:|---:|---:|---:|---:|---:|
| topology_prediction | 0.934354 | 0.780099 | 0.838944 | 0.660713 | 6.711 | 10.338 |
| prediction | 0.947435 | 0.927768 | 0.933672 | 0.891978 | 8.081 | 13.157 |
| room_labels | 0.936897 | 0.914431 | 0.922084 | 0.864208 | 8.054 | 13.031 |

| Paired delta vs topology | Node F1 | 95% CI | Edge overlap | 95% CI |
|---|---:|---:|---:|---:|
| prediction | +0.094729 | [+0.089551, +0.099799] | +0.231265 | [+0.222126, +0.240190] |
| room_labels | +0.083140 | [+0.077498, +0.088687] | +0.203494 | [+0.192961, +0.213531] |

All sources are existing frozen outputs. This audit selects no new model and uses oracle-control, non-deployable rows.
