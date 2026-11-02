# Predicted-Control Evidence Audit v1

## Correction to the Previous Reading

The oracle-control and predicted-control results are not two estimates of the
same task. Mixing them created an unnecessarily pessimistic view of the model.

| Evidence family | Control used at inference | Room IoU | Instance IoU | Valid use |
|---|---|---:|---:|---|
| Layout-conditioned completion | target-derived coarse layout | 0.9718 | 0.9711 | capacity upper bound |
| Guided MiTA-CFM | target-derived guided source | 0.7196 | not established | control-identifiability ceiling |
| DMD-v11 + room segmenter | predicted, target-free | 0.1713 | 0.5171 | strongest current deployable row |
| Predicted-control refiner | predicted, target-free | 0.0716 | not reported in this report | local refinement evidence |
| Predicted completion v1 | predicted_v1 | 0.0266 | 0.0997 | constrained-control baseline |
| Predicted guided v2 | predicted_guided_v1 | 0.0292 | 0.0772 | control-refinement screen |

The authoritative machine-readable map is
`outputs/reports/predicted_control_evidence_map_v1.json`.

## The Important Positive Signal

The strongest target-free stack is not the 0.068 legacy test row. It is the
DMD-v11 distribution model fused with the independently trained ResNet18-FPN
room segmenter:

- room macro IoU `0.17131`;
- instance geometric IoU `0.51706`;
- room support IoU `0.74833`;
- validity `1.00000`;
- watertightness `0.92445`;
- wall F1 `0.42235`;
- opening F1 `0.11450`.

The large gap between instance IoU and class-wise room IoU is the actionable
failure. Predicted geometry is often reasonably matched as an instance, but
fragmentation, duplicate masks, and identity assignment destroy the class-wise
score. The problem is therefore instance identity and semantic consolidation,
not a complete absence of geometric capability.

The 0.7196 guided result is a control cap, not a deployable score:
`scripts/evaluate_contour_cfm_pipeline.py` obtains its guided source from
`guided_source_from_labels(target, ...)`. The 0.9718 layout-conditioned result
likewise depends on target-derived coarse layout controls.

## Designs Worth Reusing

The compatible design pattern is coarse control followed by structural
refinement:

- HouseDiffusion separates discrete structural decisions from continuous
  geometry refinement.
- MaskPLAN uses partial layout masks and learned dynamic masking rather than a
  single global raster prediction.
- Raster2Seq uses explicit anchors and room polygons instead of relying on
  pixel argmax alone.
- FloorFlow already contains the same useful primitives: a contour room
  segmenter, DMD-V11, recursive room priors, room-mask decoder, coarse
  structure/furniture/support refiners, and MiTA routing.

The next model should therefore be a predicted-control bridge, not a new
backbone:

```text
footprint
  -> predicted room support and instance anchors
  -> room identity consolidation / duplicate resolution
  -> confidence-calibrated coarse control
  -> existing wall / opening / furniture / support refiners
  -> canonical CAD output
```

## Next Experiment

PC-02 freezes DMD-v11 + room segmenter as the validation baseline and tests a
room-instance identity head. The first comparison must use validation only and
must report the full chain:

- instance geometry is preserved;
- room macro IoU and instance F1 improve together;
- wall/opening metrics do not collapse;
- validity and watertightness do not regress;
- no target-derived input is read at inference.

Only a materially positive, five-seed-stable bridge should be promoted to the
full predicted-control and free-text editing paths.
