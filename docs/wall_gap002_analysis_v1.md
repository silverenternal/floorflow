# Wall Gap-Fraction Calibration v1

Date: 2026-11-02

## Diagnosis

On canonical predicted-control val491, predicted wall count has Pearson
correlation `-0.569` with wall precision but only `0.071` with wall recall.
The dominant wall error is therefore over-segmentation/spurious fragments, not
missing walls.

## Gap Fraction Screen

The prior pipeline fixed `reconcile_gap_fraction=0.04`. A matched val128
screen gives:

| Gap | Wall F1 | Precision | Recall | Wall count |
| ---: | ---: | ---: | ---: | ---: |
| .04 | 0.511185 | 0.426916 | 0.705723 | 47.516 |
| .02 | 0.523933 | 0.445203 | 0.702087 | 44.781 |
| .06 | 0.475688 | 0.382338 | 0.712783 | 55.688 |
| .08 | 0.436865 | 0.335441 | 0.725506 | 67.336 |

The `.02` arm is promoted. On full val491 its Wall F1 is `0.500706`, a paired
gain of `+0.013300` with 95% CI `[0.011968, 0.014594]` and Wilcoxon
`p=4.92e-54`. Precision rises by `0.018362`; recall falls only `0.003393`.
Room, Room IoU, Node, Edge, deterministic GED, validity, and watertightness are
exactly unchanged. Wall continuity is `0.991698`, above the `0.99` gate.

## Rejected Alternatives

IP reconcile with MILP reduces Wall F1 to `0.454883`; without MILP it leaves
wall metrics unchanged but lowers Room F1 by `0.047546`.

## Remaining Bound

Absolute Wall F1 remains `0.079294` below the `0.58` gate and precision remains
below `0.52`. The next wall work is selective fragment scoring, not another
global merge.
