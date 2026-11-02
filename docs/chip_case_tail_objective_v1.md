# Case-Tail-Aware Physics Objective v1

Date: 2026-10-31

Status: rejected at the frozen warm-start quick screen.

## Motivation

The three-seed range replication produced stable cost gains but unstable V_rel,
with seed21 dominated by a small number of high-block failures. The mean
physics objective gives those failures the same weight as ordinary cases.

## Objective

The existing per-case `batch_physics_terms` output is reused. In addition to the
mean physics and clean-physics terms, the candidate optimizes:

```text
top_fraction_mean(physics_cases, 0.25)
+ top_fraction_mean(clean_physics_cases, 0.25)
```

The tail term is weighted at `0.5`. It adds no model parameters and does not
use target-free evaluation metrics outside the existing training batch.

The implementation supports delayed ramping so a later variant can learn the
mean solution first. The first frozen screen used weight `0.5`, fraction `0.25`
and no delay.

## Result

At NFE4/refiner400 on the frozen 32-case screen:

| Metric | Control | Tail objective | Relative |
|---|---:|---:|---:|
| Cost | 1.29635 | 1.27125 | -1.94% |
| V_rel | 0.05058 | 0.05721 | +13.09% |
| Feasibility | 1.0 | 1.0 | - |

Cost median changed by only `-0.00013`; V_rel median was unchanged, with
6 improved, 10 worsened and 16 unchanged cases. The term shifts a few
high-error cases rather than reducing the structural tail.

## Decision

Reject the objective without confirmation. The default mean objective remains
unchanged. Do not tune the fraction or weight inside the same frozen screen.

## Artifacts

- `src/floorflow/chip_flow/train_objective.py`
- `tests/test_chip_train_objective.py`
- `configs/chip_tail_objective_warmstart_screen_v1.json`
- `outputs/reports/chip_tail_objective_warmstart_screen_v1.json`
