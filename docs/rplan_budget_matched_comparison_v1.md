# RPLAN Budget and Baseline Comparison v1

## CE2EPlan Anchor

The official CE2EPlan `b_model300000.pt` was run on the same 256 validation
cases and 256x256 boundary contract:

| Metric | CE2EPlan pretrained |
| --- | ---: |
| Room IoU | 0.6601 |
| Instance F1 | 0.4719 |
| Type accuracy | 0.7325 |
| Count MAE | 0.8477 |

Post-processing completed for all 256 cases with zero evaluation failures.

## FloorFlow Candidate

The current class-agnostic separation candidate at 1024 train / 256 validation
reaches:

| Metric | FloorFlow |
| --- | ---: |
| Room IoU | 0.5042 |
| Instance F1 | 0.2696 |
| Type accuracy | 0.6231 |
| Count MAE | 0.9336 |

## Budget

| Property | CE2EPlan | FloorFlow |
| --- | ---: | ---: |
| Parameters | 9.64M | 13.61M |
| Training sample exposures | about 153.6M | 8,192 |
| Optimizer steps represented by current run | 300k | 4,096 |
| Sampling NFE | 1,000 | 1 |

The architecture sizes are comparable, but the training budgets are not. The
CE2EPlan row is a pretrained external anchor and cannot be presented as a
matched-budget defeat.

## Supervised Separation Result

Using Hungarian assignments to restrict separation to same-target-class positive
queries underperforms the class-agnostic term:

| Arm | Room IoU | Instance F1 | Type accuracy |
| --- | ---: | ---: | ---: |
| Class-agnostic separation | **0.4493** | **0.1842** | **0.5299** |
| Supervised same-class separation | 0.4130 | 0.1653 | 0.2734 |

The supervised variant removes too much of the dense query interaction signal
and is rejected. No further separation-weight or affinity tuning is authorized.

## Decision

- Keep class-agnostic separation as a diagnostic candidate.
- Keep CE2EPlan as a disclosed pretrained anchor.
- Do not claim matched SOTA or efficiency.
- Move to instance-mask arbitration using matched mask evidence.
