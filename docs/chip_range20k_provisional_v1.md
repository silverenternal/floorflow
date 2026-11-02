# High-Block Range Coverage: Seed-19 Provisional Result

Date: 2026-10-30

Status: mean-level pass, paired-CI unstable, seed replication running.

## Protocol

- Candidate: frozen 20k-case subset spanning 21-120 blocks from the 200k range
  cache.
- Control: clean 20k-case subset spanning 21-60 blocks.
- Both: seed19, 20k steps, batch128, projector4, cuda training, cuda:1 dev32
  evaluation, NFE4/refiner400.

The candidate subset is balanced across the range: 7,999 cases in 21-60,
6,136 in 61-90 and 5,865 in 91-120.

## Pooled Result

| Metric | 21-60 control | 21-120 candidate | Relative |
|---|---:|---:|---:|
| Legalized cost | 1.240490 | 1.209283 | -2.52% |
| V_rel | 0.056765 | 0.051288 | -9.65% |
| Feasibility | 1.0 | 1.0 | unchanged |

The case-paired confidence intervals cross zero for both primary metrics.
Seventeen cases improve and fourteen worsen on cost.

This arm is still worse than the full-50k-cache production 20k checkpoint:
cost `+1.65%` and V_rel `+20.09%`. Against the seed19 group=2 Trinity d192
anchor it remains `+11.14%` cost and `+113.85%` V_rel. The present result proves
only that broader block coverage helps the matched subset control.

## Strata

| Blocks | Control cost / V_rel | Range cost / V_rel |
|---|---|---|
| 21-60 | 1.1752 / 0.04260 | 1.1697 / 0.04461 |
| 61-90 | 1.2761 / 0.06683 | 1.2272 / 0.05736 |
| 91-120 | 1.2933 / 0.06612 | 1.2447 / 0.05450 |
| 100-120 | 1.2477 / 0.05913 | 1.1928 / 0.04984 |

High-complexity strata improve consistently. The low-block stratum has a 0.47%
cost improvement but a 4.7% V_rel regression. Grouping improves by 37.5% with a
paired CI excluding zero. Rho/aspect MAE improves by 7.13% on the mean, but its
paired interval crosses zero.

## Decision

This is not yet a registered candidate. It is a provisional data-coverage result
whose mean gain is not robust at the case level. Seeds 20 and 21 must reproduce
both pooled metrics against their same-seed controls. Promotion additionally
requires acceptable low-block behavior and a later full-data gate.

## Artifacts

- `outputs/reports/chip_sota_training_range_control_v1.json`
- `outputs/reports/chip_range_control_comparison_v1.json`
- `outputs/reports/chip_range20k_highblock_analysis_v1.json`
- `outputs/reports/chip_range20k_coverage_v1.json`

