# S6 Legalized Group Action Space v1

Date: 2026-11-01

## Purpose

The deterministic post-legalizer group repair greedily chooses the first
shortest component translation. S6 asks whether a learned policy can choose a
better feasible action from the same legalized placement.

The action enumerator reuses the certified group-repair checks and adds no
target-derived input features. A candidate action is retained only when:

- the placement stays hard-feasible;
- boundary violations do not increase;
- the selected cluster loses at least one connected component;
- absolute b2b and p2b lengths stay inside the declared cap.

Features include move geometry, component/main sizes, graph cut weights,
boundary delta, group bbox change and cluster id. Targets, GT boxes and official
cost are not inputs.

## Dense Contact Search

The initial contact search tried only three alignment positions per edge. It
produced 9 actions on the 92-case development split and 38 actions on the
408-case train split.

The revised search enumerates all positive-length overlap alignments along each
main-block edge. With the same 2% net cap it produces:

| Split | Cases | Cases with actions | Actions | Multi-action cases |
| --- | ---: | ---: | ---: | ---: |
| Train408 | 408 | 27 | 80 | 12 |
| Development92 | 92 | 6 | 14 | 4 |

All 14 development actions satisfy the hard, boundary and net checks. Action
counts on development are: `0:86, 1:2, 2:2, 4:2`.

On the initial contact search, increasing the move cap from 25% to 50% added
only one action on train408, and increasing the net cap from 2% to 5% added no
action. The bottleneck is neither the move horizon nor the net tolerance.

## Decision

Do not train an S6 action selector on this support. In 381/408 train cases
there is no certified one-step action, and only 12 train cases offer a ranking
choice. A learned ranker cannot create new coverage from this state space.

This is not a rejection of legalizer-compatible learning. It rejects only the
current "move one split component until it touches the current main component"
action generator.

## Next Action Generator

The next S6 generator must create genuinely new feasible options:

1. rip up an unprotected component and search free-space placements adjacent
   to any cluster member, not only the selected main component;
2. repack a multi-block cluster as a rigid shelf/template around its anchor;
3. use legalizer response labels to propose bounded local reflow regions;
4. enforce the same hard, boundary and b2b/p2b safety contract.

Only after multi-action support exists on a meaningful train fraction should a
within-case scorer be trained.

## Artifacts

- `src/floorflow/chip_flow/legalized_group_actions.py`
- `tests/test_chip_legalized_group_actions.py`
- `scripts/collect_chip_legalized_group_actions.py`
- `outputs/reports/chip_s6_legalized_group_actions_train408_v2.json`
- `outputs/reports/chip_s6_legalized_group_actions_val92_v2.json`
