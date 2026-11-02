# Learned CAD Wall-Candidate Selection Review

## Accepted Zero-Training Mechanism

The room-partition wall-candidate mechanism clears the complete-plan gate on
the target-watertight calibration subset of 55/64 cases:

| Metric | Eligible-case value |
| --- | ---: |
| Vector Wall F1 | 0.3683 |
| Vector Opening F1 | 0.1211 |
| Vector Room F1 | 0.2322 |
| Schema validity | 1.0000 |
| Watertightness | 1.0000 |
| HouseMind-style Node F1 | 0.7188 |
| HouseMind-style Edge F1 | 0.4714 |
| HouseMind-style GED | 7.9091 |

The all-case gate is lower because the target itself is not watertight on 9/64
cases. Both all-case and watertight-eligible rows must remain in the evidence
record.

The accepted recipe is:

1. Predicted room cells produce junction-split wall candidates.
2. The wall layer scores each candidate along its centerline.
3. Candidate evidence at `0.2` selects walls, using the staged layer rather than
   the high-recall runtime wall probability.
4. The high-recall runtime wall skeleton is used only as the opening host plan,
   not as the final wall output.
5. Openings use the dedicated detector at threshold `0.3` and attach to the
   selected final walls.
6. Room cells below 64 pixels are removed to satisfy fpeval's region-tiling
   minimum, and only the largest predicted footprint component is tiled.

## Remaining Defect

The deterministic evidence threshold trades precision and recall with one
global rule. It cannot learn room-wall semantics, junction degree, host
continuity, or which room boundary is an open-plan edge. Wall precision is
`0.409` and recall `0.370`, leaving both false positives and missing walls.

## Next Hypothesis

A small learned candidate scorer should select and locally snap the same
candidate set without changing its identity:

- Sample staged field features at both endpoints and the midpoint.
- Feed candidate length, direction, footprint exposure, wall-layer evidence,
  and typed stage context into the scorer.
- Predict candidate existence and a bounded endpoint residual.
- Train only against canonical vector targets, but select checkpoints on the
  composed fpeval gate.
- Keep the opening source and room tiling fixed to isolate wall selection.

## Matched Control and Budget

- Control: deterministic candidate evidence threshold `0.2`.
- Treatment: identical candidates, layer checkpoint, train256/cal64 split,
  optimizer updates, and opening/room composition.
- Budget: eight epochs, batch size 4, about 512 updates, `cuda:0`.
- No new backbone and no target input at inference.
- Extend to 24 epochs only if wall F1 or host coverage improves over the final
  three checkpoints.

## Gate

On the 55 watertight-eligible calibration cases:

- Vector Wall F1 > `0.3683`;
- Vector Opening F1 >= `0.1211`;
- Vector Room F1 >= `0.2322`;
- schema validity and watertightness remain `1.0`;
- all-case schema validity remains `1.0`;
- no test access and no proxy endpoint F1 substituted for fpeval.

