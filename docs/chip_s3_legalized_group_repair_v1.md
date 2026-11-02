# S3 Legalized Group Repair v1

Date: 2026-11-01

## Mechanism

The global `scale_pack` legalizer is preserved. After legalization, a local
target-free repair visits every split cluster and may rigidly translate one
unprotected component until it shares an edge with the cluster's main
component. Fixed, preplaced and boundary-coded components never move.

Every accepted move must retain:

- hard feasibility;
- no increase in boundary violations;
- a strict reduction in the current cluster's connected components;
- absolute b2b and p2b lengths within a predeclared 2% tolerance.

The search uses no target positions, target metrics or legalizer labels.

## Fresh 500-Case Result

The frozen S3 structural selector chooses the base trajectory. Local repair
accepts 19 component moves.

| Comparison | Cost | V_rel | Grouping | Boundary |
| --- | ---: | ---: | ---: | ---: |
| Repair vs selector base | -0.196% | -2.81% | -6.31% | 0 |
| Repair vs Trinity NFE4/ref1600 | -3.89% | -17.66% | +8.88% | -26.74% |

Repair-vs-base paired intervals exclude zero favorably for cost, V_rel and
grouping. Area, rho and boundary are unchanged. Absolute b2b and p2b lengths
change by `+0.0094%` and `+0.0023%`, inside the declared 2% local tolerance.

Against the Trinity anchor:

- cost 95% CI `[-7.58%, -1.67%]`;
- V_rel 95% CI `[-0.904%, -0.217%]`;
- joint wins are `258/500`;
- strict structural wins are `115/500`, unchanged from selection-only;
- constraint-safe wins are `40/500`, unchanged from selection-only;
- grouping remains `+8.88%` and its interval crosses zero;
- rho remains `+8.38%` and is still worse.

Against the matched NFE16/ref1600 anchor:

- cost `-3.22%`, 95% CI `[-6.63%, -0.94%]`;
- V_rel `-14.00%`, 95% CI `[-0.736%, -0.107%]`;
- grouping `+5.22%`, interval crossing zero;
- boundary `-21.88%`, interval excluding zero;
- joint wins `263/500`;
- strict structural wins `121/500`;
- constraint-safe wins `50/500`.

The four-view portfolio consumes 16 total NFE, 1,600 refiner steps and four
legalizer calls. This is an absolute-budget result, not a same-single-path
inference claim.

## Decision

Promote legalized group repair as a component of the deployable candidate. It
is a safe partial repair, not a SOTA result by itself. It improves pooled
means but does not increase anchor-relative strict/constraint-safe case
coverage. The remaining gap is candidate generation for unrepairable clusters
and target-free rho.

## Artifacts

- `src/floorflow/chip_flow/legalized_group_repair.py`
- `scripts/evaluate_chip_legalized_group_repair.py`
- `tests/test_chip_legalized_group_repair.py`
- `outputs/reports/chip_s3_legalized_group_repair_val92_v5.json`
- `outputs/reports/chip_s3_legalized_group_repair_holdout500_v4.json`
- `outputs/reports/chip_s3_legalized_group_repair_holdout500_vs_nfe16_v2.json`

