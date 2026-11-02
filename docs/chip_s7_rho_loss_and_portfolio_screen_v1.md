# S7 Rho-Loss and Legalizer-Portfolio Screen

Date: 2026-11-01

## Direct Rho Loss

The matched 300-step arm changes only `rho_weight: 0 -> 0.25` on the same M27
seed21 backbone and 1k subset.

| Metric | Control | rho-loss | Relative |
| --- | ---: | ---: | ---: |
| Rho MAE | 0.124162 | 0.124017 | -0.12% |
| Grouping | 0.318092 | 0.319499 | +0.44% |
| Boundary | 0.114079 | 0.112792 | -1.13% |
| Area | 0.976885 | 0.976735 | -0.02% |
| Total | 0.016796 | 0.016793 | -0.02% |

Reject before 2k or 500-case expansion.

## Legalizer/Projector Portfolio

A fixed 92-case screen combines four candidate views with four pinned
configurations: baseline, expanded relation, restored/no-reshape and
projector32.

| Oracle | Baseline views | 16-variant portfolio |
| --- | ---: | ---: |
| JSR | 62/92 | 63/92 |
| Strict structural | 37/92 | 40/92 |
| Four-metric | 23/92 | 27/92 |
| Constraint-safe | 15/92 | 15/92 |

The extra variants add zero CSJ oracle cases. Reject this direction; legalizer
portfolio growth does not create the missing safe candidates.

## Artifacts

- `configs/chip_s7_rho_loss_v1.json`
- `outputs/chip_hierarchy/s7_rho_loss_v1_quick/rho_loss_v1.json`
- `outputs/reports/chip_s3_legalizer_projector_portfolio_val92_v1.json`
- `outputs/reports/chip_s3_legalizer_expanded_val92_v1.json`
- `outputs/reports/chip_s3_legalizer_noreshape_val92_v1.json`
- `outputs/reports/chip_s3_projector32_val92_v1.json`

