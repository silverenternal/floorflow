# Predicted-Control Structural Plausibility v1

## Scope

This is a 256/64 mechanism screen on native CubiCasa rasters. It is not a
leaderboard result and does not authorize a full-split run. Test503 remains
closed. IoU is reported only as a diagnostic.

## Metric Sources

The suite is documented in `docs/structural_plausibility_metrics_v1.md`.
`fpeval` is used directly for structural verification, wall/opening matching,
and edit cost. Raster2Seq/RoomFormer's 10 px corner and 5 degree angle
tolerances are adapted into a local deterministic row adapter because the
upstream evaluator requires dataset-specific `DataRW`. GSDiff remains the
reference for shared-edge adjacency and room-type GED; FloorFlow uses one
fixed local graph readout for paired comparisons.

## Main Finding

The raw decoded bridge is not structurally plausible even though its raw room
semantic validity flag is high. Before topology composition:

- parsable rate: `0.2344`;
- watertightness: `0.2344`;
- room closure: `0.2344`;
- self-intersection rate: `0.7656`;
- wall continuity: `0.7298`.

Composing the known exterior contour is the dominant fix. It does not change
the learned room prediction substantially, but makes the output a closed
architectural object.

## Selected Arms

| Arm | Parsable | Watertight | Closure | Self-intersect | Overlap | Wall cont. | Holes | Dangling | Wall F1 | Opening F1 | Corner F1 | Angle F1 | Edit cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Structure / exterior | 1.000 | 0.938 | 1.000 | 0.000 | 0.00016 | 0.970 | 0.031 | 0.0162 | 0.303 | 0.276 | 0.195 | 0.272 | 163.77 |
| Structure / repair | 1.000 | 0.938 | 1.000 | 0.000 | 0.00016 | 0.971 | 0.031 | 0.0155 | **0.402** | 0.186 | **0.198** | 0.262 | 194.79 |
| Structure / partition | 1.000 | 0.922 | 1.000 | 0.000 | 0.00074 | **0.988** | 0.094 | 0.0054 | 0.251 | **0.277** | **0.212** | 0.258 | 255.79 |
| Router 0.75 / exterior | 1.000 | **0.938** | 1.000 | 0.000 | 0.00016 | 0.975 | **0.000** | 0.0137 | 0.300 | 0.275 | 0.196 | **0.269** | **163.71** |

The first-64 candidate `router 0.75 / exterior` did not generalize to the
disjoint offset64 holdout: watertightness fell to `0.8906`, below the gate.
The stable candidate is instead `structure / exterior` with boundary width 7.
On the fresh offset128 holdout it reaches validity/parsability `1.0`,
watertightness `0.9375`, room closure `1.0`, zero self-intersection, zero room
overlap, wall continuity `0.9727`, and wall F1 `0.3245`.

## Decisions

- Promote exterior-contour composition as a required finalization primitive.
- Keep room repair as a wall-quality option, not a global default: it raises
  wall F1 but lowers opening F1.
- Reject partition-only walls as the main decoder: they improve wall continuity
  but produce holes, high edit cost, and worse GED.
- Keep the router as a weak structural candidate; it does not justify a full
  split because it was selected on the same 64-case validation subset used for
  branch checkpoint selection.
- Post-hoc contour vectorization is rejected as the main readout. Epsilon 3.0
  raises corner F1 from `0.195` to `0.220` and wall continuity from `0.970` to
  `0.989`, but increases edit cost from `163.8` to `228.0` and GED from `11.78`
  to `17.14`.
- A learned pixel-level boundary residual is also rejected. In isolation it
  raises Corner F1 `0.1962->0.2087` and Angle F1 `0.3321->0.3837`, but after
  the CAD bridge Room IoU remains `0.281`, Node F1 falls from `0.1410` to
  `0.1189`, and GED rises from `11.78` to `11.88`.
- A direct anchor-conditioned primitive head is rejected: despite Corner F1
  `0.231` and Angle F1 `0.451`, rasterized Room IoU is only `0.0042`, so room
  existence and identity are not learned.
- A residual refiner over predicted room proposals is a weak candidate only.
  On the disjoint holdout its best safe scale improves Corner F1
  `0.2529->0.2597`, but instance F1 falls `0.3376->0.3285`, Angle F1 falls,
  and watertightness remains `0.8438`.

## Structural Decision

Boundary width 7 plus a target-free parsable fallback is the structural safety
baseline. The full train5126/val491 run reaches:

| Metric | Value |
|---|---:|
| Structural validity | 1.0000 |
| Parsable rate | 1.0000 |
| Watertightness | 0.9124 |
| Room closure | 1.0000 |
| Self-intersection | 0.0000 |
| Room overlap | 0.000057 |
| Wall continuity | 0.9794 |
| Hole count per case | 0.0081 |
| Wall F1 | 0.3381 |
| Opening F1 | 0.3234 |
| Corner F1 | 0.1880 |
| Angle F1 | 0.4294 |

The parsable fallback is intrinsic-only and triggered on two of 491 cases; the
other 489 are unchanged. This is not a strong geometric or graph-fidelity
claim: Room IoU is `0.2704`, Node F1 is `0.1101`, and edge overlap is `0.0581`.

With the safety mechanism frozen, Room IoU and geometric fidelity return as the
next optimization target. Polygon refiners must retain this structural gate as
a hard acceptance guard.

Machine-readable evidence:
`outputs/reports/predicted_control_structural_plausibility_v1.json` and
`outputs/reports/raster_control_router_screen256_v1.json`.
