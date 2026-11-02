# Predicted-Control Room Capability Audit v1

## Evidence

The current val64 control is recorded in
`outputs/reports/predicted_control_room_gap_strata_v1.json`.

- Overall Room IoU: `0.30808`
- Instance F1: `0.21977`
- Count absolute error: `3.09375`
- Wall F1: `0.50059`
- Opening F1: `0.19688`
- Conditional watertightness: `0.48438`

Room-count strata:

| Target room count | Cases | Room IoU | Instance F1 | Count error | Wall F1 | Watertight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0-4 | 3 | 0.37705 | 0.29074 | 1.67 | 0.61862 | 0.33333 |
| 5-7 | 29 | 0.33942 | 0.23324 | 2.66 | 0.55271 | 0.65517 |
| 8+ | 32 | 0.27322 | 0.20090 | 3.63 | 0.44229 | 0.34375 |

The existing test503 decomposition provides the failure-cause split:

- semantic room identity confusion: `92.25%`
- spurious room: `3.98%`
- boundary mismatch: `3.58%`
- instance fragmentation: `0.20%`

Occupancy remains around `0.696`, while correct room identity accounts for only
`0.205` of target room pixels. The dominant bottleneck is therefore semantic
instance identity and count, not polygon boundary polish.

## Architectural Change

The new `RoomInstanceIdentityStage` reuses the existing `RoomMaskDecoder` and
adds:

1. dense room logits as decoder prototype memory;
2. room action token conditioning;
3. a zero-initialized room residual scale;
4. instance-aware room class/mask outputs;
5. a zero-initialized room-to-structure feature adapter.

The room stage is preserved for old checkpoints. The instance stage is opt-in.
At initialization the room residual and structure adapter are both identity,
so a new model starts from the current dense control before learning.

## Training Contract

- Instance query loss uses the existing Hungarian room-mask loss.
- Dense partition loss constrains the fused query masks to the target room
  partition.
- Wall and opening stages receive the room-conditioned feature adapter.
- No target-derived input enters inference.
- The first screen is matched train64/cal32 for eight epochs on `cuda:0`.

## Screen Gate

The branch advances only if Room IoU and instance F1 improve materially with
wall/opening, validity, and conditional watertightness intact. A positive
64/32 result must still pass 256/64 before val491 or multi-seed promotion.

## Screen Result

The instance-residual screen failed:

- dense Room IoU: `0.28999`
- refined Room IoU: `0.28917`
- query-only Room IoU: `0.14385`
- dense instance F1: `0.21751`
- refined instance F1: `0.21852`
- wall/opening regressions: `-0.01048 / -0.01471`

The residual scale reached only `0.01738`; the query masks were intrinsically
worse than the dense field. Raster2Seq proposal masks did not repair this, and
greedy proposal classification collapsed Room IoU to `0.02520`. The next
architecture therefore preserves official Raster2Seq polygon geometry instead
of regenerating room masks.
