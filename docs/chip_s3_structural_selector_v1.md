# S3 Target-Free Structural Selector v1

Date: 2026-11-01

## Decision

The frozen selector is a deployable internal candidate:

- fit on 408 cases from the v3/v4 cohort;
- selected on the 92-case development holdout;
- confirmed once on a disjoint fresh 500-case cohort;
- uses only target-free candidate inputs at inference;
- evaluates all four candidates and charges four legalizer calls to the budget.

It materially improves every internal metric versus candidate 0 and reduces the
Trinity gap on cost, V_rel, area, b2b, p2b, boundary and legalizer work. It does
not close the Trinity grouping/rho gap, so it is not a SOTA promotion.

## Why the Previous Selector Failed

The earlier learned selector mixed two separate contracts:

1. it used legalized V_rel, grouping, boundary and MIB as inference features,
   which require ground truth;
2. after those fields were removed, a small learned ranker trained on the
   11-column feature set overfit the legalized cost target and failed to improve
   structural quality.

The replacement separates supervision from inference. Legalized metrics are
used only as offline training labels. The deployed input is the 14-column
target-free candidate geometry contract plus within-case ranks, deltas from the
reference candidate, and the known source-view identity.

## Model

- family: `ExtraTreesRegressor` from the project's existing scikit-learn
  dependency;
- 500 trees, maximum depth 4, minimum leaf size 8;
- 46 input columns, 10 legalized metric outputs;
- 12,472 total tree nodes, 1.9 MB serialized checkpoint;
- CPU inference only;
- target-derived inference features: false.

The selector predicts legalized cost, V_rel, area, b2b, p2b, grouping, boundary,
rho and legalizer work. Its frozen utility weights emphasize cost, V_rel, rho
and grouping.

## Label Reproducibility

The parallel candidate-label evaluator reuses the existing hard projector,
exact-area decoder, official scorer and legalizer. It evaluates only real
candidates and never inserts a synthetic consensus arm.

- 500 cases and 2,000 candidate rows per cohort;
- 8 workers, one solver thread per worker;
- 26.3 seconds for the v3 cohort;
- 25.2 seconds for the fresh cohort;
- 1,472 overlapping values from the previous 92-case report matched exactly,
  maximum absolute difference `0.0`.

## Development Holdout: 92 Cases

Selector versus candidate 0:

| Metric | Relative | Paired CI |
| --- | ---: | ---: |
| cost | -8.78% | excludes zero |
| V_rel | -27.39% | excludes zero |
| area gap | -52.21% | excludes zero |
| b2b HPWL gap | -58.75% | crosses zero |
| p2b HPWL gap | -97.46% | excludes zero |
| grouping | -31.31% | excludes zero |
| boundary | -21.74% | excludes zero |
| rho MAE | -2.38% | crosses zero |

The selector is also better than the fixed `min_b2b` policy on cost (-1.64%),
V_rel (-11.57%), grouping (-4.23%), rho (-3.48%), boundary and legalizer work,
but gives back some area and b2b.

## Fresh 500-Case Confirmation

Fresh cohort versus candidate 0:

| Metric | Relative | Paired CI |
| --- | ---: | ---: |
| cost | -5.61% | excludes zero |
| V_rel | -13.71% | excludes zero |
| area gap | -41.23% | excludes zero |
| b2b HPWL gap | -52.28% | excludes zero |
| p2b HPWL gap | -77.17% | excludes zero |
| grouping | -17.53% | excludes zero |
| boundary | -12.47% | excludes zero |
| rho MAE | -3.48% | excludes zero |
| legalizer rung | -4.18% | excludes zero |
| legalizer solves | -7.25% | excludes zero |

Joint primary wins are `389/500` and constraint-safe wins are `238/500`
against candidate 0.

Versus the frozen `min_b2b` policy, the selector keeps cost flat, improves V_rel
by 3.13%, rho by 4.07%, boundary by 5.44%, rung by 5.06% and solves by 3.96%.
It deliberately gives back area (`+14.63%`) and b2b (`+47.42%`) while remaining
far better than candidate 0 on both.

## Trinity Anchors

Against Trinity d192 NFE4 / refiner1600:

| Metric | Relative |
| --- | ---: |
| cost | -3.70% |
| V_rel | -15.28% |
| area gap | -15.01% |
| b2b HPWL gap | -21.16% |
| p2b HPWL gap | -79.78% |
| grouping | +16.22% |
| boundary | -26.74% |
| rho MAE | +8.38% |
| legalizer rung | -11.48% |
| legalizer solves | -18.17% |

Cost, V_rel, p2b and boundary intervals exclude zero favorably. Area and b2b
means improve but their intervals cross zero. Rho regresses with an interval
excluding zero; grouping regresses in mean but its interval crosses zero.

Only `254/500` cases pass the cost+V_rel joint gate, `115/500` pass the strict
structural gate, and `40/500` pass the constraint-safe joint gate.

Against Trinity NFE16 / refiner1600, the direction is the same: cost -3.03%,
V_rel -11.51%, grouping +12.31%, rho +6.42%, with `257/500` primary joint wins
and `49/500` constraint-safe wins.

## Interpretation

The selector proves that the existing four-view candidate pool contains enough
signal to:

- improve legalized cost and V_rel;
- reduce p2b and boundary violations;
- reduce grouping relative to the fixed reference;
- reduce rho error relative to both the fixed reference and `min_b2b`;
- improve legalizer survival and work.

It also proves the ceiling of selection-only repair. No fixed candidate beats
the Trinity mean grouping and rho targets. The fresh candidate-pool oracle has
only `132/500` cases with any candidate simultaneously non-regressing on
cost, V_rel, grouping and rho. A selector cannot create a missing legalized
structural state.

## Stop Condition

Retain this selector as the S3 deployment candidate. Do not tune its utility
weights again on the fresh cohort. The next S3 promotion requires either:

1. a generated group-preserving candidate that survives legalization, supplied
   by the S1 group-state or S2 endpoint path; or
2. a cheap target-free group/rho repair whose exact-area and feasibility
   contracts hold before and after legalization.

Until one of those exists, do not claim Trinity grouping/rho parity or
constraint-safe FloorSet SOTA.

## Artifacts

- `scripts/evaluate_chip_candidate_legalized_labels.py`
- `scripts/train_chip_candidate_structural_selector.py`
- `src/floorflow/chip_flow/candidate_structural_selector.py`
- `outputs/reports/chip_s3_candidate_legalized_labels_v3_500_v1.json`
- `outputs/reports/chip_s3_candidate_legalized_labels_holdout500_v2.json`
- `outputs/reports/chip_s3_structural_selector_v1.json`
- `outputs/reports/chip_s3_structural_selector_v1_vs_nfe16.json`
- `outputs/train/chip_s3_structural_selector_v1/selector.pkl`
