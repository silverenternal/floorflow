# Existing Raster Outputs Under the Canonical Room Graph

| Source | Node P | Node R | Node F1 | Edge overlap | Nodes | Edges |
|---|---:|---:|---:|---:|---:|---:|
| topology_prediction | 0.932707 | 0.779784 | 0.837909 | 0.660279 | 6.715 | 10.342 |
| prediction | 0.950944 | 0.927554 | 0.934798 | 0.892946 | 8.039 | 13.072 |
| room_labels | 0.945098 | 0.914756 | 0.926231 | 0.867532 | 7.957 | 12.811 |

| Paired delta vs topology | Node F1 | 95% CI | Edge overlap | 95% CI |
|---|---:|---:|---:|---:|
| prediction | +0.096889 | [+0.091646, +0.102203] | +0.232667 | [+0.223642, +0.241875] |
| room_labels | +0.088322 | [+0.082743, +0.093815] | +0.207253 | [+0.196914, +0.217385] |

All sources are existing frozen outputs. This audit selects no new model and uses oracle-control, non-deployable rows.
