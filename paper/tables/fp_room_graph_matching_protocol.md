# Room-Graph Matching Protocol Sensitivity

| Protocol | Node P | Node R | Node F1 | Edge overlap |
|---|---:|---:|---:|---:|
| label_iou | 0.312474 | 0.600512 | 0.391224 | 0.333890 |
| geometry_hungarian | 0.347190 | 0.669663 | 0.435366 | 0.399064 |
| type_geometry_hungarian | 0.347173 | 0.669623 | 0.435341 | 0.398988 |

Unit-cost GED mean: 20.335216; exact on 1971 cases, fallback on 508.

The label-plus-IoU protocol remains primary. Alternate rows are matching-sensitivity diagnostics and must not be selected as a headline.
