# Predicted-Control Structure Budget v1

## Decision

Extending the existing coarse structure bridge from three to eight epochs does
not improve end-to-end wall/opening quality. The wall and opening curves reach a
plateau during the continuation and the best checkpoint is slightly worse than
the three-epoch checkpoint on calibration64.

## Pre-Execution Assessment

The original structure-only bridge completed three full-data epochs. Wall F1,
opening F1, and watertightness were still increasing at the final recorded
epoch, so a five-epoch continuation was a justified budget probe.

Both arms use the same train5126/val491 cases, seed, batch size, optimizer,
structure resolution, and GT structure loss. The continuation warm-starts the
three-epoch checkpoint and uses the current 384px native room checkpoint.

## Continuation Curve

| Continued epoch | Wall F1 | Opening F1 | Watertight | Room closure |
|---:|---:|---:|---:|---:|
| 1 | 0.52750 | 0.33309 | 0.39308 | 0.42770 |
| 2 | 0.52405 | 0.33488 | 0.30143 | 0.32790 |
| 3 | 0.53485 | 0.34317 | 0.41752 | 0.45418 |
| 4 | **0.53702** | 0.34497 | **0.43585** | **0.47658** |
| 5 | 0.53677 | **0.34589** | 0.41752 | 0.45621 |

The final epoch does not improve the selected checkpoint. The checkpoint at
continued epoch 4 is retained as the best continuation artifact.

## End-to-End Calibration64

The old and continued structure models are evaluated with the same safe-routing
room candidate and explicit-plan evaluator.

| Structure checkpoint | Room IoU | Instance F1 | Wall F1 | Opening F1 | Node F1 | Edge F1 | GED | Plan plausible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Three-epoch | 0.33254 | 0.27715 | 0.45847 | 0.11135 | 0.74477 | 0.49428 | 9.4688 | 0.96875 |
| Eight-epoch continuation | 0.33476 | 0.27667 | 0.45654 | 0.11037 | 0.74108 | 0.49374 | 9.4531 | 0.96875 |

The continuation trades a small Room-IoU gain for neutral-to-negative wall,
opening, and graph metrics. It is rejected as the PC-06 structure solution.

## Next Consequence

Do not continue adding epochs to the same coarse mask bridge. The next PC-06
arm must use a structure representation that exposes sub-cell wall alignment and
opening intervals, reusing the existing geometry-centerline and
opening-interval heads before adding another loss.

## Evidence

- `outputs/train/raster_structure_bridge_full_v1/best.pt`
- `outputs/train/raster_structure_bridge_full_8ep_v2/best.pt`
- `outputs/reports/room_identity_safety_global_cal64_v1.json`
- `outputs/reports/room_identity_safety_global_structure8ep_cal64_v2.json`
- `outputs/reports/geometry_heads_p6_cal64_v1.json`
