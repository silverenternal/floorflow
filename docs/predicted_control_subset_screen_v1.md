# Predicted-Control Subset Screen v1

## Policy

Mechanism confirmation now uses a 256-case training subset and a fixed 64-case
validation subset. These rows are screening evidence, not leaderboard or
deployment results. A mechanism is promoted to train5126 and val491 only after
it clears a predeclared relative gate on the subset. Test503 remains closed.

The screen is target-free at inference. Native CubiCasa PNGs are the input, and
native room-class pixels are never read by the model.

## Evidence Boundaries

The earlier oracle-control `Room IoU 0.9718` remains a capacity upper bound. It
does not measure deployable predicted-control quality. The deployable control
track has three distinct inputs:

| Track | Input/control | Allowed interpretation |
|---|---|---|
| Oracle layout | target-derived layout | capacity upper bound only |
| Clean non-room render | rendered wall/opening/furniture/structure | reconstruction with annotation-derived rendering |
| Native raster | CubiCasa PNG | target-free image reconstruction |

Rows from these tracks must not be pooled.

## Screens

All rows below use 256 training cases, 64 validation cases, three epochs, and
`cuda:0`.

| Arm | Room IoU | Instance IoU | Instance F1 | Type acc. | Wall F1 | Opening F1 | Watertight |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native dense room control | 0.1893 | 0.4798 | 0.1652 | 0.4753 | - | - | - |
| Instance-query blend 0.25 | **0.1942** | **0.5123** | **0.1758** | **0.5234** | - | - | - |
| Joint control bridge | 0.2256 | 0.5065 | **0.1930** | 0.5133 | **0.5582** | 0.2972 | 0.1875 |
| Structure-only bridge | **0.2688** | 0.5042 | 0.1709 | **0.6063** | 0.5539 | **0.3090** | **0.2344** |

The instance-query route is a weak candidate: it improves matched instance
geometry, but its Room IoU gain is only `0.0049` and instance F1 is unstable
across epochs.

The joint bridge is a candidate with a safety regression. It improves instance
F1 and learns usable wall/opening structure, but watertightness collapses.

The structure-only bridge is the current room-preservation candidate. Keeping
the native room prediction fixed avoids the destructive room-refinement step
and gives the best Room IoU and type accuracy in this screen.

## Why The Shared Full-CAD Bank Is Not Promoted

The first shared room/object encoder screen used a larger subset and did not
produce calibrated opening or furniture foregrounds. Its structure metrics
therefore came from uninformative dense outputs rather than a usable control.
That route is retained as a negative mechanism result and is not scheduled for
train5126.

## Next

1. Add a validation-only router that selects structure-only or joint-bridge
   room evidence from confidence and proposal agreement.
2. Apply the existing vector topology projection after structure decoding and
   require watertight recovery without changing validity.
3. Re-run only the router/topology arm on the same 256/64 screen.
4. Promote to train5126 and val491 only if Room IoU, instance F1, wall/opening
   F1, validity, and watertightness all clear their subset gates.

Machine-readable evidence:
`outputs/reports/predicted_control_subset_screen_summary_v1.json`.
