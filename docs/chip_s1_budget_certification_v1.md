# S1 Named-Budget Certification

Date: 2026-11-01

## Decision

The M75 finalist passes one named budget class only:

- absolute budget: pass;
- equal inference: fail;
- equal-quality efficiency: not established;
- external routed PPA: not run.

Do not describe the result as same-inference SOTA or as a cheaper path to the
same quality. The correct statement is:

> Under a fully disclosed 10-trajectory, NFE14, refiner16000 and 15-legalizer
> absolute budget, target-free seed-diverse selection beats three Trinity d192
> anchors in mean cost and V_rel with feasibility 1.0, while retaining a
> material area regression and no external PPA certification.

## Absolute Budget

| Quantity | M75 finalist | Trinity d192 anchor |
|---|---:|---:|
| Parameters | 6,797,217 | 5,856,400 |
| Candidate relative parameter count | +16.06% | baseline |
| Trajectories | 10 | 1 |
| Total NFE | 14 | 4 |
| Total refiner steps | 16,000 | 3,200 |
| Legalizer calls | 15 | 1 |
| Official-100 feasible fraction | 1.0 | 1.0 |

The candidate has fewer fresh training exposures because it is initialized from
the Trinity backbone. That does not offset the larger inference/search budget.

## Quality Pareto

| Candidate / anchor | NFE | Refiner steps | Legalizer calls | Cost | V_rel |
|---|---:|---:|---:|---:|---:|
| M48 portfolio seed19 | 2 | 3,200 | 3 | 1.080161 | 0.016695 |
| M74 fixed-seed cross-checkpoint | 10 | 9,600 | 9 | 1.050223 | 0.009877 |
| M75 seed-diverse guarded | 14 | 16,000 | 15 | 1.039928 | 0.009770 |
| Trinity seed19 | 4 | 3,200 | 1 | 1.052969 | 0.020741 |
| Trinity seed20 | 4 | 3,200 | 1 | 1.045964 | 0.017752 |
| Trinity seed21 | 4 | 3,200 | 1 | 1.070648 | 0.021616 |

M75 is the only retained candidate point below the strongest anchor mean on
both primary metrics. It reaches that point after M48 and M74 fail the joint
quality threshold.

## Equal-Quality Reading

M75 meets the quality region, but it uses more NFE, refiner work and legalizer
calls than the anchor. Therefore:

- it is not an equal-quality efficiency win;
- it is not a same-inference win;
- it is an absolute-budget quality result.

The seed-diverse search is useful for rescuing a catastrophic high-complexity
tail, but its search cost must be reported as part of the method.

## Guards

| Guard | Result |
|---|---|
| Feasibility | pass, 1.0 |
| Solver escalation | pass, zero |
| Cost | pass against all three anchor means |
| V_rel | pass against all three anchor means |
| Area | fail; M75 remains worse |
| b2b | fail on mean guard |
| External routed WL/congestion | not run |
| DRC/timing/power/area | not run |

Area and external PPA are the remaining blockers before a CVPR SOTA claim.
M76/M56 target the area/cost tail. E1 remains mandatory for routed-PPA transfer.

## Artifacts

- `outputs/reports/chip_s1_named_budget_certification_v1.json`
- `outputs/reports/chip_s1_equal_quality_pareto_v1.json`
- `scripts/build_chip_s1_named_budget_certification.py`
