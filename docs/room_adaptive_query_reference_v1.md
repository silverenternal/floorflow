# Room Adaptive Query Reference v1

Date: 2026-11-01

## Finding

`AdaptiveQueryReferenceRefiner` existed and was wired into the model, but
`--class-only` recovery kept it frozen. No saved training report exercised the
module in the canonical room-head path.

## Repair

- The recovery control now unfreezes the adaptive reference sampler when it is
  enabled.
- Tests assert that its parameters are included in class recovery.
- Existing checkpoints still disable it by default.

## Matched Screen

Both arms use the same parent, 64-train/32-validation subset, seed, and
four-epoch class-only budget.

| Arm | Room IoU | Instance F1 | Type | Count MAE | Selection |
|---|---:|---:|---:|---:|---:|
| static query | .26593 | .26274 | .67188 | 3.9063 | .01830 |
| adaptive reference | .26519 | .26824 | .64955 | 3.8125 | .01759 |
| program + adaptive | .26530 | .27852 | .63802 | 3.9063 | .01700 |

Decision: reject promotion. Adaptive sampling improves instance F1 and count,
but regresses Room IoU, type, and the balanced selection score. Combining it
with program conditioning raises F1 further but loses more type accuracy and
selection. It is useful as an instance-geometry mechanism, not as the main
semantic-identity repair.

Evidence:

- `outputs/train/room_adaptive_static_fast64_v1/report.json`
- `outputs/train/room_adaptive_query_fast64_v1/report.json`
- `outputs/train/room_program_adaptive_combo_fast64_v1/report.json`
- `outputs/reports/room_adaptive_query_screen_fast64_v1.json`
- `outputs/reports/room_program_adaptive_combo_screen_fast64_v1.json`

## Formal 256/64 Screen

The predeclared static and program+adaptive arms were repeated on 256 training
cases and 64 validation cases:

| Arm | Room IoU | Instance F1 | Type | Count MAE | Selection |
|---|---:|---:|---:|---:|---:|
| static query | .26206 | .24420 | .60998 | 3.4688 | .01914 |
| program + adaptive | .26007 | .23675 | .60503 | 3.4688 | .01706 |

The candidate regresses all three quality dimensions except count, so the
combination is closed. The formal evidence is:

- `outputs/train/room_adaptive_static_screen256_v1/report.json`
- `outputs/train/room_program_adaptive_combo_screen256_v1/report.json`
- `outputs/reports/room_program_adaptive_combo_screen256_v1.json`

