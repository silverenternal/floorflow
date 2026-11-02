# TopoGED to FloorFlow Predicted-Control Mapping

## Source

- Paper: *A Topological Encoder Decoder Framework for Temporal Graph Learning*.
- Venue: NeurIPS 2026.
- Paper page: `https://neurips.cc/virtual/2026/poster/151161`.
- OpenReview: `https://openreview.net/forum?id=cZEj9pgOP7`.
- Official repository: `https://github.com/FDataLab/TopoGED`.
- Local sparse clone: `third_party/TopoGED`.
- Pinned commit: `fde9441dc666c0dac5036f8d88cd46c190ad6e49`.

The upstream repository is a sparse clone because the full history/data is
close to 1 GB. The checked-out files cover the descriptor, end-to-end graph
construction, candidate prediction, graphlet loss, and TGN utilities.

## Relevant Mechanism

TopoGED does not predict every edge independently. It first predicts:

1. a multiscale filtration descriptor;
2. node and edge budgets;
3. edge-type proportions;
4. an arrival/reappearance distribution;

and only then reconstructs the future graph in phases under those budgets.
Its `GraphletLoss` combines distribution MSE, histogram intersection, and
cumulative L1 (Wasserstein-style) distance.

## FloorFlow Mapping

| TopoGED | FloorFlow predicted-control analogue |
| --- | --- |
| Graph snapshot | Room-partition wall-candidate graph |
| Node budget | outer + interior wall-segment budget |
| Edge budget | junction-incidence budget |
| Multiscale filtration | node/induced-link counts over junction-degree thresholds |
| Edge-type phases | footprint wall, room-to-room wall, room-boundary wall, opening host |
| Candidate link score | wall-candidate existence logit |
| Arrival memory | room/boundary candidate source and prior-stage evidence |
| Graphlet loss | filtration-descriptor MSE + intersection + CDF loss |
| Phased reconstruction | top-K wall budget, then host-wall openings, then validation |

## Implemented Probe

`CadWallCandidateScorer` now predicts:

- per-candidate wall existence;
- a bounded global wall budget;
- bounded endpoint residuals.

`candidate_filtration_loss` computes endpoint-junction adjacency and an
eight-dimensional multiscale descriptor: node and induced-link counts at
junction-degree thresholds `1, 2, 3, 4`. Training combines the descriptor loss
with per-candidate BCE, wall-budget regression, and matched endpoint geometry.

Inference reconstructs the wall layer by selecting the top predicted wall
budget rather than applying a dataset-wide score threshold. This implements the
paper's budget-first reconstruction principle in the residential CAD layer.

## Boundaries

- TopoGED solves temporal graph snapshots; FloorFlow reconstructs a static CAD
  plan from predicted controls.
- The current descriptor operates on the candidate segment graph, not on a
  learned temporal node memory.
- The floorplan method must still be validated by the existing composed fpeval,
  validity, and watertightness gates; descriptor loss alone is not evidence.

