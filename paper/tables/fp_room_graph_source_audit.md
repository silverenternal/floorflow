# Existing Raster Outputs Under the Canonical Room Graph

| Source | Node P | Node R | Node F1 | Edge overlap | Nodes | Edges |
|---|---:|---:|---:|---:|---:|---:|
| topology_prediction | 0.934120 | 0.777623 | 0.837135 | 0.657761 | 6.687 | 10.275 |
| prediction | 0.952122 | 0.931319 | 0.937616 | 0.897818 | 8.076 | 13.161 |
| room_labels | 0.945835 | 0.918161 | 0.928534 | 0.871396 | 7.989 | 12.863 |

| Paired delta vs topology | Node F1 | 95% CI | Edge overlap | 95% CI |
|---|---:|---:|---:|---:|
| prediction | +0.100481 | [+0.095104, +0.105728] | +0.240057 | [+0.230871, +0.249316] |
| room_labels | +0.091399 | [+0.085808, +0.096841] | +0.213635 | [+0.202923, +0.224062] |

All sources are existing frozen outputs. This audit selects no new model and uses oracle-control, non-deployable rows.
