# Room Program Conditioning v1

Date: 2026-11-01

## Finding

`RasterInstanceControl` exposed room-program logits but never connected the
program class tokens to the query class head. The query decoder was also built
without `program_token_dim`, so passing tokens could not have worked.

## Repair

- Added backward-compatible `program_token_conditioning`.
- Program class tokens now feed `RoomMaskClassFusion` when the flag is enabled.
- The flag requires `room_program_head` and defaults off, preserving old
  checkpoints and behavior.
- Class-recovery training now includes `room_program_head` parameters.
- Checkpoint loading restores the conditioning flag.
- Tests verify the attention path and program-head gradients.

## Matched Screen

Both arms start from the same seed19 query parent and use the same 64-train /
32-validation subset, four epochs, class-only recovery, and program loss `0.2`.

| Arm | Room IoU | Instance F1 | Type | Count MAE | Support IoU | Selection |
|---|---:|---:|---:|---:|---:|---:|
| supervision only | .26493 | .25022 | .66302 | 3.5000 | .76073 | .01917 |
| direct class prior | .26346 | .25975 | .65521 | 3.3438 | .76073 | .01832 |
| token conditioning | .26579 | .25766 | .67039 | 3.8750 | .76073 | .01829 |
| full-head conditioning | .26286 | .26369 | .66615 | 3.7813 | .76073 | .01574 |

Decision: reject promotion. Conditioning raises Room IoU, instance F1, and type
slightly, but count MAE regresses by `0.375` and selection score falls. Letting
mask geometry adapt lowers selection further. The program pathway was not wired
before; the consumer is now implemented, but this budget does not satisfy the
joint identity gate. Keep it as an opt-in ablation.

The parameter-free direct class prior gives the opposite trade: F1 rises
`0.00953` and count improves `0.15625`, while Room IoU and type regress. This
confirms the program signal is informative but not calibrated strongly enough
to promote.

Evidence:

- `outputs/train/room_program_supervision_only_fast64_v2/report.json`
- `outputs/train/room_program_conditioning_fast64_v2/report.json`
- `outputs/train/room_program_conditioning_full_fast64_v2/report.json`
- `outputs/train/room_program_class_prior_fast64_v1/report.json`
- `outputs/reports/room_program_conditioning_screen_fast64_v1.json`
- `outputs/reports/room_program_class_prior_screen_fast64_v1.json`

