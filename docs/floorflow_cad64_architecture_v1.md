# Compact Partition-Edge CAD Architecture v1

## Motivation

The RPLAN target wall count averaged `27.79` over 4,096 training cases (p95
`36`, p99 `40`, maximum `47`). The previous 128-query wall bank therefore spent
most existence supervision on queries that could never correspond to a wall.

The compact CAD arm uses:

- 64 wall queries;
- four opening queries;
- fixed-cardinality Hungarian supervision;
- dynamic bounded positive weights; and
- optional partition-edge evidence computed from competing room-mask logits.

## Matched 1024/256 Screen

Seeds `2905` and `37`, 12 epochs, batch 8, one-forward NFE.

| Metric | Compact control | Partition edge | Mean delta |
| --- | ---: | ---: | ---: |
| Room IoU | 0.64950 | 0.64939 | -0.00010 |
| Instance F1 | 0.47761 | 0.48678 | +0.00917 |
| Type accuracy | 0.71272 | 0.71569 | +0.00297 |
| Count MAE | 0.91211 | 0.90234 | -0.00977 |
| Node F1 | 0.88949 | 0.93113 | +0.04163 |
| Edge overlap | 0.56586 | 0.63611 | +0.07025 |
| GED | 3.76953 | 3.08008 | -0.68945 |
| Conditional watertightness | 0.99805 | 0.99023 | -0.00782 |
| Raw wall F1 | 0.39526 | 0.40621 | +0.01096 |

Validity is `1.0` in every run. Node F1, edge overlap, and GED improve on both
seeds. Raw wall F1 is seed-sensitive (`+0.0643` on seed 2905, `-0.0424` on seed
37), so it is not promoted as a robust main metric.

## 4096 Screen

Seed 2905, train4096/validation512, eight epochs.

| Metric | Compact control | Partition edge |
| --- | ---: | ---: |
| Room IoU | **0.66108** | 0.65797 |
| Instance F1 | **0.50686** | 0.48368 |
| Type accuracy | **0.76090** | 0.75607 |
| Count MAE | 0.94531 | **0.91406** |
| Node F1 | 0.90003 | **0.90893** |
| Edge overlap | 0.60175 | **0.60641** |
| GED | 3.27734 | **3.22852** |
| Conditional watertightness | 0.99023 | 0.99023 |
| Raw wall F1 | **0.41967** | 0.40392 |

At larger scale, edge conditioning no longer improves F1 and therefore cannot
be the general identity architecture. It remains a topology-specialized
variant. The compact control proceeds to a five-seed quality run.

## Learned Openings

The learned opening head remains near-zero F1 (`0.0017-0.0034` in the tested
4096 arms). Canonical opening F1 `1.0` comes from the supplied entrance
condition and is never reported as learned opening capability.

## Decision

- Promote the 64-query mask-only control for quality experiments.
- Keep partition-edge conditioning as a graph/topology diagnostic.
- Reject the combined entrance+edge arm from the main architecture. Across
  seeds 2905 and 37 it raises mean type accuracy to `0.72897` and raw wall F1
  to `0.44033`, but lowers mean Room IoU to `0.64716`, F1 to `0.47288`, Node F1
  to `0.91731`, and edge overlap to `0.61829`; GED rises to `3.29297`. All graph
  results are worse than the edge-only topology arm.
- Do not connect the learned opening head to the main output.

