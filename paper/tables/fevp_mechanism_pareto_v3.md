# Audited Mechanism Pareto v3

| Audited arm | Violation | Delta-IoU | Node F1 | Edge F1 | GED | Frontier |
|---|---:|---:|---:|---:|---:|---|
| Typed | 0.000000000 | 0.937473 | 0.955068 | 0.890007 | 2.120855 | yes |
| Dense | 0.000455107 | 0.887741 | 0.953797 | 0.887080 | 2.185478 | no |
| No gate | 0.013884758 | 0.853715 | 0.949057 | 0.874121 | 2.448649 | no |
| No graph | 0.000000000 | 0.888817 | 0.953819 | 0.887200 | 2.182251 | no |

## Un-audited Context

| Method | Delta-IoU | Node F1 | Edge overlap | GED | Locality axis |
|---|---:|---:|---:|---:|---|
| HouseMind-E | 0.608 | 0.998 | 0.934 | 0.467 | not reported |
| HouseMind-O | 0.598 | 0.997 | 0.908 | 0.653 | not reported |
| HouseMind-G | — | — | — | — | not reported |

Among audited arms, Typed routing is the joint locality-quality frontier and dominates No graph on Delta-IoU, Node F1, Edge F1, and GED. HouseMind-E/O cannot be plotted on the locality axis because the public evaluation does not report outside exactness or violation rate; they appear only as un-audited context.

The current fixed-evaluator rows show Typed better than No graph on Delta-IoU, Node F1, Edge F1, and GED. The older visual reading that Typed only wins locality was caused by an incorrect frontier membership calculation; the corrected table and figure are authoritative.

HouseMind-G is generation-only and cannot enter an editing Pareto.
