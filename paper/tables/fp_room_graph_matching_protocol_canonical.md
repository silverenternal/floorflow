# Room-Graph Matching Protocol Sensitivity

| Protocol | Node P | Node R | Node F1 | Edge overlap |
|---|---:|---:|---:|---:|
| label_iou | 0.934120 | 0.777623 | 0.837135 | 0.657761 |
| geometry_hungarian | 0.943207 | 0.783756 | 0.844104 | 0.661599 |
| type_geometry_hungarian | 0.942804 | 0.783675 | 0.843970 | 0.661599 |

Unit-cost GED mean: 6.326341; exact on 2232 cases, fallback on 247.

The label-plus-IoU protocol remains primary. Alternate rows are matching-sensitivity diagnostics and must not be selected as a headline.
