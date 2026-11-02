# Existing Raster Outputs Under the Canonical Room Graph

| Source | Node P | Node R | Node F1 | Edge overlap | Nodes | Edges |
|---|---:|---:|---:|---:|---:|---:|
| topology_prediction | 0.931015 | 0.775202 | 0.834664 | 0.655096 | 6.675 | 10.262 |
| prediction | 0.947709 | 0.925188 | 0.931733 | 0.888994 | 8.058 | 13.136 |
| room_labels | 0.945364 | 0.914468 | 0.926105 | 0.867197 | 7.954 | 12.843 |

| Paired delta vs topology | Node F1 | 95% CI | Edge overlap | 95% CI |
|---|---:|---:|---:|---:|
| prediction | +0.097069 | [+0.091693, +0.102336] | +0.233899 | [+0.224883, +0.242681] |
| room_labels | +0.091441 | [+0.085879, +0.096924] | +0.212102 | [+0.201897, +0.222156] |

All sources are existing frozen outputs. This audit selects no new model and uses oracle-control, non-deployable rows.
