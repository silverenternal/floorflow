# Native Room High-Resolution Budget Study v1

## Decision

Training resolution and budget materially improve target-free native-raster room
identity. The 384px model is promoted as the new room checkpoint for integration;
it is not yet a SOTA result.

## Matched Subset Screen

All arms use the same warm start, train256/cal64, seed 2903, class balancing,
augmentation, optimizer, and loss.

| Resolution | Budget | Room IoU | Instance F1 | Type accuracy |
|---|---:|---:|---:|---:|
| 256 | 8 epochs | 0.20238 | 0.16363 | 0.49557 |
| 384 | 8 epochs | 0.20995 | 0.17839 | 0.51295 |
| 384 | 24 epochs | 0.20995 | 0.18667 | 0.54857 |

The three target metrics improve together, so the mechanism is retained.

## Full-Split Capacity Probe

| Model | Room IoU | Instance F1 | Type accuracy |
|---|---:|---:|---:|
| 256 full baseline | 0.25919 | 0.20498 | 0.57230 |
| 384, 8 epochs | 0.311998 | 0.238635 | 0.655587 |
| 384, 24 epochs | 0.319722 | 0.253162 | 0.660906 |
| 384, 32 epochs | 0.323570 | 0.258259 | 0.673713 |

The 32-epoch model is the selected room checkpoint. The model remains below the
eventual RC-03B `.35/.30` room promotion gate, but it is a substantial and
consistent improvement over the prior room-identity baseline.

## End-to-End Predicted-Control Check

The promoted room checkpoint is integrated with the existing predicted
structural bridge and explicit-room legalizer.

| Split | Room IoU | Instance F1 | Type accuracy | Validity | Functional watertight | Wall continuity | Plan plausible |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cal64 | 0.33138 | 0.27067 | 0.68411 | 1.0 | 1.0 | 0.98991 | 0.96875 |
| Holdout128 | 0.39594 | 0.31783 | 0.81328 | 1.0 | 0.98438 | 0.99341 | 0.96875 |
| Val491 | 0.33386 | 0.28883 | 0.68362 | 1.0 | 1.0 | 0.98856 | 0.95519 |

Calibration and disjoint holdout agree in direction and both preserve the
reasonableness gates. Full val491 preserves the gains, but narrowly misses the
predeclared room gate of Room IoU 0.35 and instance F1 0.30. The remaining gap is
an instance-identity/cardinality problem rather than a reasonableness failure:
count MAE is 3.57026 and the model over-predicts rooms.

The 40-epoch continuation was stopped after its first five validation epochs
peaked at Room IoU 0.31839, below the selected 32-epoch value. Generic fragment
merging with minimum area 64 changes no calibration64 room metric. These results
rule out "more epochs" and "merge fragments" as the next main moves.

## Boundary

- The 384px model uses native CubiCasa raster input and no room-label input.
- The model is selected on validation only.
- Test503 remains closed.
- Improved room logits alone do not justify a SOTA claim; full-CAD and graph
  metrics must be reported from the same checkpoint.
