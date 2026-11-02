# Adaptive Model Audit v1

The complete learning-head wiring and verification are recorded in
`docs/room_head_optimization_v2.md`.

## Scope

This audit checks whether feature construction, query routing, and multi-loss
optimization are genuinely sample-adaptive in the promoted predicted-control
path. A dynamic module that exists only in an unused branch does not count.

## Findings

| Area | Previous promoted behavior | Dynamic modules already present | Status |
| --- | --- | --- | --- |
| Room queries | static query embeddings plus masked multi-head cross-attention | `MixtureTopKAttention`, `SparseControlAttention` exist in other branches | not dynamic in the promoted decoder |
| Spatial feature access | dense decoder tokens and adaptive pooling | no learned reference sampling in the room-query path | static grid |
| Support gate | dense/query probabilities, entropy, agreement, query support | learned conv deletion residual | sample/spatially conditioned |
| Stage routing | ordered typed graph with learned gates | not the causal runtime of the current predicted-CAD evaluator | partially adopted |
| Loss weights | fixed room-mask, partition, dense, and teacher weights | `GradientNormBalancer` exists in CFM branches | now opt-in for room-query training |
| Opening host | heatmap sampled on a wall plan | detector threshold and host distance fixed | typed consumer, threshold-fixed |

## Repairs Implemented

### Image-conditioned adaptive query references

`AdaptiveQueryReferenceRefiner` learns sample-dependent reference points from
the room query embedding and image-global feature, samples the feature map with
`grid_sample`, and returns a residual query update. Its output projection is
zero-initialized, so enabling the module starts from the parent model behavior.

The model exposes reference points, displacement, and sampled-feature norm as
diagnostics. It does not mutate the canonical room labels until trained.

### Adaptive room-loss weights

`train_raster_instance_control.py` can now route room-mask, partition,
dense-identity, and optional teacher losses through the existing
`GradientNormBalancer`. Scales are EMA-smoothed, bounded, recorded per epoch,
and stored in checkpoints. The default remains the fixed-weight objective.

### Disabled-path contract

- Existing checkpoints do not instantiate the adaptive refiner.
- New refiner output is zero at initialization.
- Dynamic loss weights are opt-in.
- Tests cover reference shape/range, gradient flow, module wiring, and disabled
  checkpoint compatibility.

## Mathematical References

The following open-source references were cloned under `third_party/` and used
to validate the mechanism, not copied into runtime dependencies:

| Reference | Commit | Relevant mechanism |
| --- | --- | --- |
| [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR) | `11169a60c33333af00a4849f1808023eba96a931` | sample-dependent reference points and learned sampling offsets |
| [Mask2Former](https://github.com/facebookresearch/Mask2Former) | `9b0651c6c1d5b3af2e6da0589b719c514ec0d69a` | masked transformer decoding and point-sampled mask supervision |
| [LibMTL](https://github.com/median-research-group/LibMTL) | `4336804847eaa5e0b924b743d76beec7ac3fdc97` | GradNorm, uncertainty, Nash, DWA, and gradient-conflict weighting families |

FloorFlow keeps a lightweight local implementation so inference does not depend
on Detectron2 or LibMTL.

## Required Ablation

The next screen uses one fixed data/seed/readout budget:

1. static query + fixed loss;
2. adaptive query + fixed loss;
3. static query + adaptive loss;
4. adaptive query + adaptive loss.

Promotion requires joint Room IoU/F1/count gains on calibration and untouched
holdout. A gain that only appears in aggregate training loss or one metric is
not accepted.

