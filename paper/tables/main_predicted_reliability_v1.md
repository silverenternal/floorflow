# Main Predicted-Control Reliability v1

Status: candidate table. Model and held-out test are not frozen.

## Safety Contract

| Evidence | Current result |
|---|---:|
| Supported prompts executed | 438 / 438 |
| Unsupported or ambiguous prompts abstained | 300 / 300 |
| Outside pixel exact | 1.000000 |
| Outside room-label exact | 1.000000 |
| FP-Eval validity | 1.000000 |

The latest 738-prompt report used the test manifest for development. Its legacy
summary mislabeled the split as val191; the prompt rows and control root show
the run was test. It is not an unopened held-out result.

## Structural Plausibility

CubiCasa val491, five seeds:

| Metric | Mean |
|---|---:|
| Raster plausible | 0.98493 |
| Functionally watertight | 0.99796 |
| Wall continuity | 0.99543 |
| Plan plausible | 0.98656 |
| Raw fpeval watertight | 0.81670 |

## Semantic Operation Success

Preliminary stale-model val191:

| Operation | Success rate | Denominator |
|---|---:|---:|
| room add | 0.92 | 100 |
| room type | 0.55 | 100 |
| room remove | 0.20 | 100 |
| opening delete | 0.5085 | 59 |
| furniture delete | 0.00 | 15 |

Safety execution does not imply semantic success. These rows must be recomputed
after the predicted model is frozen.

## Fidelity Limitation

Predicted-control Room macro IoU is `0.26972`. Node F1 is `0.66020`, edge
overlap is `0.26462`, and GED is `11.60081`. Fidelity remains a separate
limitation, not part of the safety claim.
