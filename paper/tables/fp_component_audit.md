# FP Component Audit

| Component | Parameters | GMACs | Latency (ms) | Decision | Matched evidence |
|---|---:|---:|---:|---|---|
| room | 479,928 | 7.77 | 2.27 | retain | room_macro_iou +0.2173 [+0.2089, +0.2259]; room_adjacency_f1 +0.5061 [+0.4459, +0.5666] |
| structure | 501,933 | 3.23 | 1.98 | retain | fpeval_wall_f1 +0.0110 [+0.0042, +0.0178]; fpeval_opening_f1 +0.0254 [+0.0148, +0.0358] |
| furniture | 506,460 | 7.41 | 2.93 | retain | furniture_iou +0.1185 [+0.1118, +0.1249] |
| support | 498,657 | 7.09 | 2.94 | replace with room-label support | room_support_iou +0.1691 [+0.1557, +0.1812] |

Intervals are paired case-level bootstrap intervals for full minus the same-checkpoint component-off path.
