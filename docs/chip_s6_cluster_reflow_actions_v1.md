# S6 Cluster Reflow Actions v1

Date: 2026-11-01

## Motivation

The initial legalized group-repair action space moved one whole component to a
contact position. It produced only 27 action cases on the 408 training split
and 6 on the 92 validation cases, even after dense edge-contact enumeration.
Increasing the move or wirelength cap did not repair the action-support gap.

## Composite Reflow

`enumerate_cluster_reflow_actions` keeps fixed, preplaced and boundary-coded
blocks stationary. Within each split cluster it starts from a protected main
component and attaches unprotected blocks one at a time to the current main
component. Intermediate moves may preserve component count; the accepted
composite action must strictly reduce the cluster's connected components.

Every intermediate and final state must retain hard feasibility, a
non-increasing boundary-violation count, and absolute b2b/p2b lengths within a
2% cap. Action features are target-free; official metrics are stored only as
offline training labels.

## Action Support

| Split | Cases | Cases with actions | Actions | Max/case |
| --- | ---: | ---: | ---: | ---: |
| train 408 | 408 | 27 | 80 | 22 |
| dev 92 | 92 | 11 | 15 | 3 |
| fresh 500 | 500 | 56 | 83 | 3 |

The composite generator raises dev action-case support from 6 to 11 and yields
83 safe actions on the fresh cohort.

## Frozen Policy

The deployed selector is target-free and lexicographic:

1. minimum absolute b2b length delta;
2. minimum absolute p2b length delta;
3. minimum normalized move distance.

On dev92, it changes anchor wins from `45/22/7` to `46/24/7`
(`joint/strict/safe`), while pooled V_rel improves from `-4.49%` to `-11.33%`
and grouping from `+7.94%` to `-11.11%`.

On the fresh 500-case split:

| Quantity | Selector base | Reflow |
| --- | ---: | ---: |
| Joint wins | 257 | **269** |
| Strict structural wins | 118 | **130** |
| Constraint-safe wins | 49 | **53** |
| Cost vs NFE16 anchor | -3.03% | **-3.53%** |
| V_rel vs NFE16 anchor | -11.51% | **-18.94%** |
| Grouping vs NFE16 anchor | +12.31% | **-9.70%** |
| Area vs NFE16 anchor | -13.57% | -13.57% |
| b2b vs NFE16 anchor | -17.28% | -17.28% |
| p2b vs NFE16 anchor | -75.36% | -75.36% |
| Boundary vs NFE16 anchor | -21.88% | -21.88% |

The reflow-versus-base paired intervals exclude zero favorably for cost,
V_rel and grouping. The b2b and p2b intervals cross zero.

## Learned Action Reranker

An ExtraTrees multi-output action-delta model was fit on 64 train actions and
tested on 15 dev actions. It selected actions on the same 11 dev cases as the
fixed policy and produced identical `46/24/7` joint/strict/safe wins. There is
no learned-model gain, so the fixed target-free policy is retained.

## Boundary

The policy improves coverage but remains below the SOTA gate:
constraint-safe wins are `53/500` versus the target `175/500`. It does not
change area or rho, and target-free rho remains the dominant unresolved
dimension. Do not train a learned reranker until held-out data contain enough
within-case alternatives.

## Artifacts

- `src/floorflow/chip_flow/legalized_group_actions.py`
- `src/floorflow/chip_flow/legalized_cluster_reflow.py`
- `scripts/collect_chip_legalized_group_actions.py`
- `scripts/collect_chip_cluster_reflow_actions.py`
- `scripts/summarize_chip_cluster_reflow_policy.py`
- `tests/test_chip_legalized_group_actions.py`
- `tests/test_chip_legalized_cluster_reflow.py`
- `outputs/reports/chip_s6_cluster_reflow_actions_val92_v1.json`
- `outputs/reports/chip_s6_cluster_reflow_actions_holdout500_v1.json`
- `outputs/reports/chip_s6_cluster_reflow_val92_vs_nfe16_v1.json`
- `outputs/reports/chip_s6_cluster_reflow_holdout500_vs_nfe16_v2.json`
- `outputs/reports/chip_s6_cluster_reflow_learned_rerank_val92_v1.json`

