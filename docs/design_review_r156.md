# r156 design review — CVPR gap, borrowed ideas, and retraining plan

## What the frozen test503 still shows

| Metric | test503 | Owner module | Root cause |
|---|---:|---|---|
| room macro IoU | 0.1703 | `distribution/pixel_head.py` + room mask decoder | dense pixel head under-separates small rooms |
| furniture IoU | 0.0603 | `heads/dynamic_object_instance.py` | rare-class queries starved; no vector-space supervision yet trained |
| fpeval opening F1 | 0.1193 | `geometry/opening_readout.py` | opening heatmap is under-calibrated (interval F1 ~0.035 vs label 0.10) |
| fpeval room polygon F1 | 0.1614 | `heads/polygon_loop.py` | polygon head not retrained with diff_ras |
| fpeval wall F1 | 0.4196 | `geometry/wall_graph_readout.py` | fragmented skeleton; re-noding degrades clean rasters |
| unconditional watertight | 0.9225 | `geometry/reconciliation.py` | generic unary-union re-noding, not edge-continuity aware |
| instance geometric IoU | 0.5160 | `evaluation/room_metric_suite.py` | instance matching + room faces |

The r154 report was generated before the final r154 tree, so its furniture
IoU (0.0690) is stale; the current tree reproduces 0.0603. Every new claim
must be regenerated from this tree.

## Web research (this round)

Three references are relevant to our exact pain points:

- Boundary-Constrained Diffusion for Floorplan Generation (arXiv
  `2602.01949`): a Boundary Cross-Attention (BCA) module improves boundary
  adherence; it also warns that FID can miss diversity collapse. No public
  repository was found. Borrow the BCA idea into
  `model/footprint_cross_channel.py`.
- CAGE, Continuity-Aware edGE Network (arXiv `2509.15459`, NeurIPS 2025,
  repo `ee-Liu/CAGE`, MIT + Commons Clause): a native edge-centric wall
  representation that yields watertight, topologically valid room boundaries.
  Cloned to `third_party/CAGE`; design reference only (no source vendored).
- Trinity (arXiv `2610.04957`, already in `third_party/Trinity`):
  differentiable constraint functions used as training loss, refiner energy,
  and scoring. Reinforces using our legalization/closure terms as trainable
  constraints rather than post-hoc loops.

## Design review verdicts

1. Boundary adherence: the contour condition is injected through the U-Net
   encoder and a zero-init cross-channel prior. Add an explicit BCA-style
   boundary-query path and make it the primary contour injector (not a
   residual).
2. Wall topology: replace generic `unary_union` re-noding in
   `geometry/reconciliation.py::_noded_segments` with an edge-continuity
   snap that only closes small gaps at junction clusters; never split a
   clean wall.
3. Opening readout: train a small opening-interval head on the decoder
   heatmap instead of thresholding an under-calibrated map; keep the label
   fallback and fusion guard.
4. Furniture: combine rare-class query recycling with the differentiable
   `furniture_vector_loss` in the training objective; no discriminator in
   the inference graph.
5. CFM convergence: token minibatch OT is wired and default; measure its
   effect in the retraining run.

## Borrow contract for the new references

| Reference | License | Contract |
|---|---|---|
| Boundary-Constrained Diffusion `2602.01949` | paper only | Reimplement BCA in `model/footprint_cross_channel.py`; cite the paper. |
| CAGE `ee-Liu/CAGE` @ `11cf0d9` | MIT + Commons Clause | Read only; reimplement edge-centric readout in `geometry/`. Do not vendor source. |
| Trinity `Kohaku-Lab/Trinity` | MIT | Already adapted; extend the differentiable-constraint framing to wall closure. |
