# Predicted-Control Fidelity Screen v1

## Scope

All experiments here are mechanism screens. Inference does not read target room
pixels, validity labels, or target geometry. Test503 remains closed.

## Results

| Mechanism | Cases | Room IoU | Instance F1 | Watertight | Corner F1 | Angle F1 | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Direct anchor-conditioned primitive head | 64 | 0.0042 | 0.0020 | - | 0.231 | 0.451 | reject |
| Proposal residual, calibration scale 0.25 | 64 | 0.2763 | 0.2126 | 0.9219 | 0.2145 | 0.1812 | weak candidate |
| Proposal residual, disjoint holdout scale 0.25 | 64 | 0.3600 | 0.3285 | 0.8438 | 0.2597 | 0.2763 | reject for promotion |
| Clean-control reprojection, pure external room model | 64 | 0.0084 | 0.0478 | 0.9375 | 0.1261 | 0.4656 | reject |
| Native 384px, warm start from 256px native model | 64 | 0.1858 | 0.2044 | - | - | - | reject for short screen |
| Pretrained ConvNeXt-Tiny FPN | 64 | 0.1064 | 0.1375 | - | - | - | reject for short screen |
| ConvNeXt-Tiny, 8 extra epochs | 64 | 0.1380 | 0.1102 | - | - | - | reject |
| Class-balanced native fine-tune | 64 | 0.2029 | 0.1718 | - | - | - | reject on holdout |

The direct anchor head can produce plausible local corner geometry but does
not learn room existence or identity. The proposal residual has a small
calibration gain but fails to reproduce a balanced improvement on disjoint
holdout data. Reprojecting the predicted structure into the clean-control
domain is also invalid: the clean-raster room model collapses from Room IoU
around `0.2801` at native control to `0.0084` after pure reprojection.

A stored Raster2Seq predicted-control diagnostic was also evaluated. Its
parsable rate and watertightness are zero, and the underlying renderer is a
project adapter rather than the official upstream preprocessing. It is retained
only as a negative adapter audit and cannot support a matched external claim.

## Decision

Do not promote any of these as the fidelity solution. The next experiment must
test either a materially stronger native-image room-control architecture or a
clearly labeled Raster2Seq proposal hybrid. A pure resolution increase does not
help in the short screen: the 384px warm-start model reaches only `0.1858` Room
IoU after four epochs. ConvNeXt-Tiny reaches `0.1380` after 12 total epochs.
Class-balanced fine-tuning reaches `0.2029` on calibration but falls to
`0.2180` on holdout versus a `0.2498` baseline. Any candidate must continue
through the frozen `structure/exterior width7 + parsable fallback` structural
gate.
