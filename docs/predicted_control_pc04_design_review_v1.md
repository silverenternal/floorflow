# Predicted-Control PC-04 Design Review v1

## Decision

Reject query-mask distillation as the next main path. The mechanism is
connected and trainable, but it trades instance identity for Room IoU and does
not beat a matched control.

## Matched Evidence

All arms use train256, calibration64, batch size 8, 64 queries, the same native
raster encoder, the same target losses, and seed 2905.

| Arm | Epochs | Room IoU | Instance F1 | Type accuracy | Composite |
|---|---:|---:|---:|---:|---:|
| 64-query, no teacher | 8 | 0.18766 | 0.19794 | 0.52604 | 0.28975 |
| 64-query, teacher 0.5 | 8 | 0.19273 | 0.17718 | 0.52474 | 0.28950 |

The teacher arm raises Room IoU by `0.00507` but loses instance F1 by `0.02076`.
The distillation loss is active: matched proposal IoU rises from `0.054` to
`0.221` across training. The failure is therefore an objective conflict with
target mask and cardinality supervision, not a disconnected auxiliary loss.

The teacher class outputs remain unusable: val491 type accuracy is `0.17435`.
Raster2Seq class labels are not copied.

## Literature Refresh

The 2026 fpvec-lab readout study compares coordinate-sequence decoding and
dense detection on the same trained network. On real CubiCasa scans, detection
is better on every wall measure, the advantage grows with plan size, and
deterministic fusion of detection and sequence outputs raises wall F1 by seven
points. This rejects a sequence-only teacher as the next mandatory direction.

PolyRoom introduces room-aware query initialization and uniform polygon
sampling to avoid non-polygonal sequences. CAGE introduces continuity-aware
edges and a dual-query transformer with denoising. The next FloorFlow model
should combine those ideas around the existing raster encoder:

1. initialize room queries from dense room-support evidence and footprint;
2. use room-level and corner-level queries;
3. supervise ordered polygons with differentiable rasterization, boundary,
   Chamfer, validity, cardinality, and graph terms;
4. retain detection-first dense evidence and deterministic reconciliation.

The 2026 topology-preserving polygon augmentation paper is also relevant for
future data augmentation, but it is not a substitute for the decoder fix.

## Budget Policy

A one-epoch run is integration-only. A three-epoch run is trend-only. A
mechanism is not rejected or promoted before eight epochs or a documented
validation plateau. Promotion uses 24 epochs or a predeclared compute-matched
budget.

## Next Task

`PC-05` builds a detection-first room-aware two-level polygon decoder. It must
first beat the matched no-polygon control and the frozen calibration64 dense
baseline in both Room IoU and instance F1. Holdout64 is not run until that
trend gate passes.

## Sources

- fpvec-lab readout study, 2026: <https://arxiv.org/abs/2608.25608>
- PolyRoom, 2024: <https://arxiv.org/abs/2407.10439>
- CAGE, NeurIPS 2025: <https://arxiv.org/abs/2509.15459>
- Topology-preserving polygon augmentation, 2026: <https://arxiv.org/abs/2603.14764>
