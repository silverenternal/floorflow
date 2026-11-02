# Native Room 512px Budget Probe v2

## Decision

The matched 512px arm is rejected. It improves two room-head indicators but
substantially worsens instance F1, count error, edge consistency, and GED at the
same 8-epoch budget. No 24-epoch or full-split run is authorized.

## Interface Audit

The first 512px cache was discarded because it inferred a 64-class taxonomy
from the screen subset, while the 384px checkpoint expects a 65-class taxonomy
including `room:Elevated`. Rebuilding with the canonical
`outputs/cache/cubicasa5k_full_v2_384/taxonomy.json` produces an identical
taxonomy SHA-256:

`d93cad02280a8c7aea50aec3c8cd025bbaea48350a54f60c8e1b2029070a499c`

The final screen cache has 256 train and 491 validation floor caches at
512x512. It preserves all 491 validation cases so calibration and offset
evaluation use the same ordered source as the 384px arm.

## Matched Room-Head Screen

Both arms warm-start from the same 384px/32-epoch checkpoint and use seed 2903,
train256, calibration64, eight epochs, batch size eight, the same augmentation,
class balance, optimizer, and decode.

| Resolution | Best epoch | Room IoU | Instance F1 | Type accuracy |
|---|---:|---:|---:|---:|
| 384 | 6 | 0.24630 | 0.21771 | 0.58099 |
| 512 | 6 | 0.24986 | 0.20693 | 0.61535 |
| Delta | - | +0.00356 | -0.01078 | +0.03436 |

The screen fails because instance F1 regresses while count/identity is the
primary bottleneck.

## End-to-End Predicted-Control Check

| Resolution | Room IoU | Instance F1 | Count MAE | Type accuracy | Node F1 | Edge F1 | GED | Plan plausible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 control | 0.31452 | 0.25055 | 2.70313 | 0.63307 | 0.74236 | 0.48219 | 8.8906 | 0.92188 |
| 512 probe | 0.30249 | 0.21741 | 6.70313 | 0.62292 | 0.70030 | 0.39794 | 13.5781 | 0.96875 |

The 512 arm predicts 14.3125 instances against 8.2969 targets, an over-count of
6.7031. Structural packing remains valid, but the instance prediction is
unusable as a fidelity improvement.

## Budget Consequence

The 512px branch consumed its predeclared eight-epoch mechanism budget and
failed the matched end-to-end gate. Keep the 384px/32-epoch checkpoint as the
fidelity frontier and reallocate the promotion budget to the type-conditioned
instance-affinity route. Do not extend 512px merely to improve Room IoU while
instance identity collapses.

## Evidence

- `outputs/train/native_room_384_budget_control_v2/report.json`
- `outputs/train/native_room_512_screen256_8ep_v2/report.json`
- `outputs/reports/predicted_control_384_budget_control_cal64_v2.json`
- `outputs/reports/predicted_control_512_screen64_v2.json`
- `outputs/cache/cubicasa5k_screen256_512_v2/manifest.json`
