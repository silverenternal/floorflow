# Existing Raster Outputs Under the Canonical Room Graph

| Source | Node P | Node R | Node F1 | Edge overlap | Nodes | Edges |
|---|---:|---:|---:|---:|---:|---:|
| topology_prediction | 0.931478 | 0.770399 | 0.831758 | 0.647891 | 6.637 | 10.186 |
| prediction | 0.945851 | 0.919574 | 0.928125 | 0.877700 | 8.015 | 13.019 |
| room_labels | 0.941848 | 0.909741 | 0.921861 | 0.859158 | 7.946 | 12.777 |

| Paired delta vs topology | Node F1 | 95% CI | Edge overlap | 95% CI |
|---|---:|---:|---:|---:|
| prediction | +0.096367 | [+0.091081, +0.101754] | +0.229808 | [+0.220946, +0.239025] |
| room_labels | +0.090103 | [+0.084465, +0.095595] | +0.211266 | [+0.200993, +0.221327] |

All sources are existing frozen outputs. This audit selects no new model and uses oracle-control, non-deployable rows.
