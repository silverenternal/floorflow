# S3 Seed21 Subset Fresh500 Decision

Date: 2026-11-01

## Validation Winner

The validation-only audit selected the budget-matched four-view set:

- M48 seed19 baseline;
- M48 seed21 retained;
- M61B projector32;
- M65 baseline.

The set replaces seed20 and keeps total NFE16/refiner1600.

## Fresh500 Result

Against Trinity NFE16/ref1600:

| Metric | Relative |
| --- | ---: |
| Cost | -1.01% |
| V_rel | -7.89% |
| Area gap | -2.55% |
| b2b | +0.96% |
| p2b | -53.95% |
| Grouping | +20.15% |
| Boundary | -20.51% |
| Rho MAE | +5.19% |

Cost and V_rel paired intervals both cross zero. Joint wins are `266/500`,
strict structural wins `120/500`, and constraint-safe wins `38/500`.

The four-metric candidate-pool oracle reaches only `157/500 = 31.4%`, below
the `35%` coverage gate even with oracle selection.

## Decision

Reject the seed21 replacement and close selector/subset tuning. The validation
gain did not survive fresh data, and the candidate pool ceiling remains below
the required coverage. Further work must create new legalizer-compatible
candidates through S6, not reweight existing views.

## Artifacts

- `docs/chip_s3_candidate_subsets_val92_v1.md`
- `outputs/reports/chip_s3_candidate_subsets_val92_v1.json`
- `outputs/reports/chip_s3_seed21_subset_holdout500_vs_nfe16_v1.json`
- `outputs/reports/chip_s3_candidate_legalized_labels_holdout500_seed21_subset_v1.json`

