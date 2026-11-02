# M56 Legal-Parent Reflow

Date: 2026-11-01

## Mechanism

M56 re-noises only high-pressure blocks around a legal parent:

1. Select at most 20% of movable blocks using active b2b net span and p2b pin
   pressure.
2. Exclude fixed and preplaced blocks.
3. Re-encode the legalized parent boxes into the exact-area latent.
4. Re-noise only the selected block state.
5. Integrate the existing hierarchy flow.
6. Restore every noncritical block to the parent after every step and after
   the final hard projection.

The implementation reuses the existing hierarchy sampler, hard projector,
target-free candidate selector and legalizer. It does not add a second flow
model.

## Locality Smoke

An initial smoke used the pre-legalizer refined state as the parent and failed:
noncritical displacement was `0.34-2.29` normalized units. The repair is to
re-encode the legalized parent boxes before reflow. After that:

- noncritical displacement is exactly `0`;
- hard-feasible fraction is `1.0`;
- solver escalation remains zero.

## Dev32 Screen

Two predeclared budgets were evaluated:

| Arm | NFE | Refiner | Legalizer calls | Cost | V_rel |
|---|---:|---:|---:|---:|---:|
| M48 seed19 portfolio | 2 | 3,200 | 3 | 1.171839 | 0.029304 |
| M56 one-step, one parent | 2 | 1,600 | 3 | 1.128979 | 0.026944 |
| M56 two-step, selected parents | 4 | 3,200 | 4 | 1.087996 | 0.019528 |
| M75 seed-diverse finalist | 14 | 16,000 | 15 | 1.094234 | 0.024441 |

The one-step arm is a cheaper candidate but does not reach M75 quality. The
two-step arm has better dev means than M75 at much lower search and refiner
budget. Its cost difference to M75 is not statistically resolved on 32 cases,
so it currently serves as a budget-frontier candidate pending Official-100.

Against M48 dev32, the two-step arm improves:

- cost `-7.15%`;
- V_rel `-33.36%`;
- area gap `-41.37%`;
- b2b HPWL gap `-54.81%`;
- p2b HPWL gap `-91.09%`;
- boundary `-40.54%`.

Grouping changes by `-13.33%`, while rho/aspect MAE is essentially unchanged.

## Claim Boundary

The dev result supports a local reflow mechanism, not final SOTA. Official-100
confirmation, multi-seed replication and external routed PPA remain mandatory.
Noncritical locality is measured before scale-pack legalization; the final
legalizer may still move blocks to resolve changed critical blocks.

## Official-100 Result

The two-step arm completed all 100 cases with feasibility `1.0`, zero solver
escalation and noncritical displacement `0`.

Relative to its own M48 baseline:

- cost `-2.16%`, paired CI excluding zero;
- V_rel `-30.61%`, paired CI excluding zero.

However, the correct comparator is the retained M48/M61B/M65 portfolio:

- cost regresses `+1.93%`;
- V_rel regresses `+5.53%`;
- against Trinity d192, cost regresses `+4.56%`;
- V_rel improves `-15.06%`, but the interval crosses zero.

The 2-step arm therefore does not replace M75 and does not advance to
multi-seed or external confirmation. The one-step cheap arm also fails the M48
portfolio comparator.

M56 is retained as a reusable local-repair operator and as evidence that
critical-block reflow alone cannot solve the structural portfolio gap.

## Artifacts

- `src/floorflow/chip_flow/legal_parent_reflow.py`
- `configs/chip_m56_legal_parent_reflow_v1.json`
- `outputs/reports/chip_m56_smoke2_single_v1.json`
- `outputs/reports/chip_m56_single_parent_equal_budget_dev32_v1.json`
- `outputs/reports/chip_m56_equal_budget_dev32_v1.json`
- `outputs/reports/chip_m56_decision_v1.json`
- `outputs/reports/chip_m56_extra_budget_decision_dev32_v1.json`
