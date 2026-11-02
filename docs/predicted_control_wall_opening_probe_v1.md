# Predicted-Control Wall / Opening Probe v1

## Decision

The existing wall-centerline model is useful evidence, but direct overlay fusion
into the room label raster is rejected as the final structure path. It raises
wall precision while damaging opening and instance consistency. The next arm
must keep room cells and wall/opening primitives as separate layers until the
canonical CAD decoder.

## Budget Preflight

The existing coarse structure bridge was first extended from three to eight
full-data epochs. The wall/opening curve plateaued and end-to-end calibration64
did not improve, so further epochs on the same mask refiner are rejected.

## Wall-Centerline Evidence

The reusable geometry-centerline model reports on calibration64:

| Metric | Value |
|---|---:|
| Wall-centerline precision | 0.59573 |
| Wall-centerline recall | 0.49865 |
| Wall-centerline F1 | 0.53704 |

Direct centerline fusion variants produce:

| Variant | Room IoU | Instance F1 | Wall precision | Wall recall | Wall F1 | Opening F1 | Node F1 | Edge F1 | GED |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Safe-routing baseline | 0.33254 | 0.27715 | 0.35730 | 0.71449 | 0.45847 | 0.11135 | 0.74477 | 0.49428 | 9.469 |
| Consensus 0.8, dilation 1 | 0.33479 | 0.27703 | **0.70170** | 0.45215 | 0.47031 | 0.09073 | 0.74038 | 0.49434 | 9.828 |
| Consensus 0.8, dilation 1, opening-protect 1 | 0.33427 | 0.27373 | 0.62325 | 0.51322 | **0.48780** | 0.10272 | 0.74108 | 0.49034 | 9.891 |
| Structured-object opening overlay | 0.33510 | 0.27206 | 0.67276 | 0.47177 | 0.47904 | 0.05325 | 0.74069 | 0.49178 | 9.938 |
| Existing host-wall interval readout, threshold 0.2 | 0.33254 | 0.27715 | 0.35730 | 0.71449 | 0.45847 | 0.11262 | 0.74477 | 0.49428 | 9.469 |
| Existing host-wall interval readout, threshold 0.3 | 0.33254 | 0.27715 | 0.35730 | 0.71449 | 0.45847 | 0.11178 | 0.74477 | 0.49428 | 9.469 |
| Existing host-wall interval readout, threshold 0.5 | 0.33254 | 0.27715 | 0.35730 | 0.71449 | 0.45847 | 0.11192 | 0.74477 | 0.49428 | 9.469 |
| Standalone centerline wall layer + interval readout | 0.33254 | 0.27715 | 0.26349 | 0.35086 | 0.28441 | 0.00240 | 0.74477 | 0.49428 | 9.469 |

The high-precision wall threshold clears the wall-precision target, but no
variant simultaneously preserves opening F1, instance F1, and graph metrics.
The structured-object opening head improves its internal pixel metric but does
not survive explicit-plan opening readout.

The existing host-wall interval reader also changes opening F1 by only
`+0.00127` at best, far below the required `0.30`. Its opening source is not
sufficiently calibrated for interval prediction.

Using the geometry-centerline prediction as a standalone wall plan is not
sufficient: fpeval wall F1 drops to `0.28441` and opening F1 to `0.00240`.
Wall evidence must augment a validated wall plan rather than replace it.

## Consequence

1. Keep wall-centerline evidence as a wall-layer prior.
2. Do not replace room pixels merely to express a wall centerline.
3. Keep room polygons, walls, and openings as separate typed layers through the
   explicit CAD conversion.
4. Train a host-wall-aware opening interval decoder against GT intervals rather
   than thresholding the existing opening-center probabilities.
5. Re-evaluate wall/opening metrics only after the canonical CAD conversion.

## Evidence

- `outputs/reports/geometry_heads_p6_cal64_v1.json`
- `outputs/reports/wall_centerline_consensus_t080_cal64_v1.json`
- `outputs/reports/wall_centerline_consensus_t080_protect1_cal64_v1.json`
- `outputs/reports/structure_evidence_wall08_opening05_cal64_v1.json`
- `docs/predicted_control_structure_budget_v1.md`
