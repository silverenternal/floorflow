# FloorFlow Owner-Wall Action Selector v1

## Purpose

This stage turns the wall stage token into a concrete action over merged owner
1-cells. It replaces the rejected global FiLM path with per-candidate
keep/delete scoring and a canonical wall consumer.

## Architecture

```text
owner cell complex
  -> merged shared-wall candidates
  -> learned-wall evidence
  -> CadWallCandidateScorer
       wall-stage token projection
       encoded candidate score
       endpoint residual head
       exterior head
  -> maximum-spanning-forest selection
  -> forced exterior candidates
  -> canonical wall override
  -> room polygons remain unchanged
```

The selector does not modify room polygons or owner adjacency. Any selected
wall set therefore cannot change room validity or room-only watertightness.

## Engineering Fixes

The initial scorer arms were invalid for owner candidates:

1. `continuity=1` saturated the evidence logit and made every candidate score
   approximately 1.
2. The original score head used 12 scalar features and ignored the encoded
   candidate token, so positive and negative wall candidates were inseparable.
3. Independent endpoint residuals moved shared junction endpoints separately
   and destroyed wall continuity.
4. Threshold-only deletion removed graph bridges and raised dangling endpoints
   above 0.05.

The corrected implementation:

- uses endpoint-matched learned-wall evidence with a 0.5 neutral prior;
- adds an opt-in encoded candidate score while keeping old checkpoints
  compatible;
- keeps canonical geometry from original owner endpoints;
- selects a maximum spanning forest before adding threshold-positive edges;
- forces exterior candidates to remain.

## Two-Seed 128/64 Screen

Seeds `2905` and `37`, 128 train / 64 validation, eight epochs, one NFE.

| Metric | Control+fusion | Owner action+fusion | Delta |
| --- | ---: | ---: | ---: |
| Room IoU | 0.45214 | 0.45214 | 0.00000 |
| Instance F1 | 0.19688 | 0.19688 | 0.00000 |
| Node F1 | 0.85334 | 0.85334 | 0.00000 |
| Edge overlap | 0.40796 | 0.40796 | 0.00000 |
| GED | 4.46094 | 4.46094 | 0.00000 |
| Wall F1 | 0.51440 | **0.51578** | **+0.00139** |
| Wall IoU | 0.33926 | **0.33976** | **+0.00050** |
| Validity | 1.0 | 1.0 | 0.0 |
| Watertightness | 1.0 | 1.0 | 0.0 |
| Dangling endpoint rate | 0.00884 | 0.00960 | +0.00075 |
| Opening F1 | 1.0 | 1.0 | 0.0 |

The direction is positive on both seeds, but the effect is small. Connector
length is 4.9-8.3% of total wall length, so promotion requires the 1024/256
screen and a stricter topology/efficiency check.

## 1024/256 Promotion

The 1024/256 screen did not preserve the gain:

| Seed | Control Wall F1 | Action Wall F1 | Control Wall IoU | Action Wall IoU |
| --- | ---: | ---: | ---: | ---: |
| 2905 | 0.671997 | 0.671015 | 0.469555 | 0.468643 |
| 37 | 0.661043 | 0.660361 | 0.459599 | 0.458671 |

Both Wall F1 and Wall IoU regress on both seeds. Room, graph, validity,
watertightness, and opening metrics remain unchanged because the selector only
changes wall geometry.

## Decision

- Pass the two-seed 128/64 mechanism gate.
- Reject the 1024/256 promotion arm.
- Keep the branch opt-in and do not alter the default shared-owner path.
- Do not make a main paper claim from this screen.
- Move the active effort to predicted-control room identity and count.

Evidence:
`outputs/reports/floorflow_owner_wall_action_promotion1024_v1.json`.
