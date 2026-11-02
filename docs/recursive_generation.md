# Learned recursive floorplan generation

FloorFlow treats recursion as a predicted action rather than a fixed decoder
rule. The inference path uses only the footprint and an optional train-bank
style token.

## Complexity policy

`floorflow.recursion.complexity` extracts eight footprint-only geometric
features: area, boundary density, bounding-box fill, aspect, perimeter,
corner density, convexity, and radial dispersion. Train-only room/opening/
furniture statistics generate direct, recursive, and split labels from
train-set complexity quantiles. `RecursionPolicy` learns the final decision.

## Divide and conquer

`floorflow.recursion.partition` builds a binary partition tree along contour
projections. The predicted recursion strength controls tree depth. Leaves are
disjoint, cover the footprint, and carry local masks and bounding boxes.
Independent local logits are merged with a direct full-image anchor. A
zero-blend limit therefore reduces exactly to direct decoding.

## Coarse-to-fine flow

`floorflow.recursion.inward_flow` defines a moving contour band and stage
weights ordered room -> wall/structure -> opening/furniture. The condition
training path can enable this with:

```bash
--contour-inward-curriculum --stage-curriculum
```

`--flow-preset multi_stage_ot` retains the multiflow token-transport coupling.
`scripts/audit_inward_curriculum.py` records the stage order and band coverage.

## Evaluation

`scripts/evaluate_recursive_room_segmenter.py` compares direct, recursive, and
policy-hybrid room decoders. `scripts/evaluate_distribution_overlay.py`
accepts `--recursion-policy-checkpoint` for the full CAD overlay suite. The
policy is selected on validation and never receives target labels at inference.
