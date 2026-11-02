# Room-Graph Matching Protocol Sensitivity

| Protocol | Node P | Node R | Node F1 | Edge overlap |
|---|---:|---:|---:|---:|
| label_iou | 0.952122 | 0.931319 | 0.937616 | 0.897818 |
| geometry_hungarian | 0.970146 | 0.942765 | 0.950833 | 0.906437 |
| type_geometry_hungarian | 0.970021 | 0.942646 | 0.950711 | 0.906479 |

Unit-cost GED mean: 1.701493; exact on 2180 cases, fallback on 299.

The label-plus-IoU protocol remains primary. Alternate rows are matching-sensitivity diagnostics and must not be selected as a headline.
