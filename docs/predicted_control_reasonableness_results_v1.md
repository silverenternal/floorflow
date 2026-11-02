# Predicted-Control Reasonableness Results v1

## Standalone FloorFlow

The frozen explicit-room readout was evaluated on all 491 validation cases over
five seeds. It is target-free at inference.

| Metric | Five-seed mean |
|---|---:|
| Raster plausible | 0.98493 |
| Room-face retention | 0.99565 |
| Validity | 1.00000 |
| Functional watertightness | 0.99796 |
| Wall continuity | 0.99543 |
| Plan plausibility | 0.98656 |

The standalone predicted-control track therefore clears the reasonableness
gate. Its remaining weakness is target-comparative room identity, not invalid
or structurally implausible output.

## Raster2Seq-Proposal Hybrid

This row is not standalone FloorFlow. It keeps Raster2Seq room proposals and
applies the FloorFlow seam, opening, exterior, and wall-gap legalizer.

| Metric | Val503 v6 |
|---|---:|
| Official Room IoU | 0.63228 |
| Official semantic Room F1 | 0.64711 |
| Official Corner F1 | 0.59368 |
| Official Opening F1 | 0.34026 |
| Adapter instance F1 | 0.83585 |
| Adapter type accuracy | 0.58870 |
| Adapter count error | 1.24851 |
| Raster plausible | 0.99006 |
| Validity | 1.00000 |
| Functional watertightness | 0.98807 |
| Wall continuity mean | 0.99480 |
| Wall-continuity case pass | 0.98608 |
| Plan plausibility | 0.97416 |

The bounded endpoint repair reduces plan failures from 26 to 13 and clears the
reasonableness gate. It changes no official Raster2Seq metric and no case-level
wall-area fraction.

The `gap=0.06` candidate exactly preserves official Raster2Seq Room, semantic
Room, Corner, and Opening values and preserves adapter fidelity. It increases
mean wall segments from `93.78` to `113.19` without changing wall-area
fraction. Bounded open-endpoint repair then lowers mean wall segments to
`112.95`. A `gap=0.08` 20-case screen did not recover the isolated failing case
and increased mean wall segments to `126.45`; it is rejected.

## Decision

Freeze the standalone reasonableness row and the clearly labeled hybrid row.
Do not chase Room IoU until the 2026 proposal-teacher route is separated from
the target-free standalone student.

The remaining fidelity bottlenecks are:

- room instance identity, count, and type;
- wall precision;
- opening recall;
- room-type graph consistency;
- standalone reproduction of the proposal identity available to the hybrid.

The next executable mechanism is proposal-teacher distillation into the
native FloorFlow query decoder, followed by an anchor-guided polygon fallback.
