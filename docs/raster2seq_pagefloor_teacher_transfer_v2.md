# Raster2Seq Page-Floor Teacher Transfer v2

Date: 2026-11-01

## Root Cause

Many CubiCasa `F*_scaled.png` files are page-level composites containing multiple
storeys. The old teacher preparation stretched the whole page into a 256 square
and whitened a floor mask whose SVG coordinates need not match the composite
pixel frame. This produced the observed teacher Room IoU of `0.06213`.

The corrected path has two independent pieces:

1. detect target-free floor drawing regions on the page with an ink-component
   fallback;
2. crop the FloorFlow target and footprint to the selected floor before
   resizing/padding both sides to 256.

The implementation is opt-in through `--page-floor-crop`; the default remains
the established full-page protocol until the crop protocol clears its promotion
gate.

## Corrected Teacher Evidence

The executable Raster2Seq checkpoint was run on 64 train cases with cropped
non-room structural control. The target was used only for the audit:

- cases: `64`
- Room support IoU: `0.8000`
- instance geometric IoU: `0.8096`
- instance F1: `0.5361`
- count MAE: `3.2344`
- class accuracy: `0.1211`
- teacher quality gate: passed

Teacher class outputs are explicitly untrusted. Training now defaults
`--teacher-class-weight` to `0.0`; class distillation requires an explicit
opt-in. This also fixes the old report field that incorrectly claimed class
labels were copied.

Evidence:

- `outputs/reports/raster2seq_teacher_train64_pagefloor_control_v1_audit.json`
- `outputs/cache/raster2seq_teacher_train64_pagefloor_control_v1/train/manifest.json`

## FloorFlow Screen

The static parent, page-floor warm start, and page-floor teacher arm were
reported on the same crop val64 split:

| Arm | Room IoU | Instance F1 | Count MAE | Type acc. |
|---|---:|---:|---:|---:|
| static parent | 0.1545 | 0.0899 | 8.47 | 0.3529 |
| crop + warm start | 0.2028 | 0.1359 | 9.78 | 0.5402 |
| crop + warm + teacher | 0.2039 | 0.1239 | 9.47 | 0.4983 |

The crop path improves geometry and type accuracy over the static parent, but
query count remains too high. Teacher distillation slightly reduces count
relative to the warm crop arm while reducing F1/type, so it does not clear the
joint promotion gate.

Evidence:

- `outputs/reports/room_pagefloor_crop_static_parent_cal64_v1.json`
- `outputs/reports/room_pagefloor_crop_warm_last_cal64_v2.json`
- `outputs/reports/room_pagefloor_crop_teacher_last_cal64_v1.json`
- `outputs/reports/room_pagefloor_crop_static_parent_holdout64_v1.json`
- `outputs/reports/room_pagefloor_crop_warm_holdout64_v2.json`
- `outputs/reports/room_pagefloor_queryset_best_cal64_v1.json`
- `outputs/reports/room_pagefloor_queryset_best_holdout64_v1.json`
- `outputs/reports/room_pagefloor_screen256_seed19_paired_v1.json`

## Crop Candidate Holdout

At the fixed blend-0.5 readout, crop + parent warm-start + encoder adaptation
improves all four primary room signals on disjoint holdout64:

| Arm | Room IoU | Instance F1 | Count MAE | Type acc. |
|---|---:|---:|---:|---:|
| static parent | 0.1861 | 0.1163 | 8.03 | 0.4805 |
| crop + warm start | 0.2144 | 0.1425 | 7.13 | 0.5872 |

The ungated query-set arm raises calibration geometry but count regresses on
holdout. Its count prediction was trained without the query-count selection
consumer, so this arm is superseded rather than promoted. Teacher distillation
still adds no joint gain over the warm crop arm.

## Paired 256-Case Screen

The train256 candidate improves geometry in both splits, but count is not
stable:

| Split | Room IoU delta | F1 delta | Count MAE delta | Type delta |
|---|---:|---:|---:|---:|
| calibration64 | +0.0560 [0.0335, 0.0799] | +0.0320 [0.0032, 0.0590] | +1.03 [-0.55, 2.66] | +0.1665 [0.0595, 0.2811] |
| disjoint holdout64 | +0.0427 [0.0109, 0.0750] | +0.0318 [0.0005, 0.0629] | -0.88 [-2.34, 0.53] | +0.0641 [-0.0406, 0.1742] |

Positive count delta means larger absolute error. Geometry and F1 gains are
paired-significant, but count regresses on calibration and its holdout gain has
a CI crossing zero. The crop arm therefore remains a geometry-stage candidate,
not a promoted end-to-end room mechanism.

## Count-Gated Query Candidate

The earlier query-set arm had trained a count head but did not consume its count
selection mask in room fusion. Enabling `query_count_gate` activates the
consumer. Two train256 seeds then reproduced positive directions on calibration
and disjoint holdout:

| Seed / blend | Split | Room IoU delta | F1 delta | Count MAE delta | Type delta |
|---|---|---:|---:|---:|---:|
| seed19 / 0.25 | cal64 | +0.0482 | +0.0509 | -1.31 | +0.1544 |
| seed19 / 0.25 | holdout64 | +0.0310 | +0.0365 | -2.89 | +0.0281 |
| seed23 / 0.75 | cal64 | +0.0495 | +0.0951 | -2.83 | +0.2109 |
| seed23 / 0.75 | holdout64 | +0.0184 | +0.0557 | -2.73 | +0.1211 |

Room IoU/F1/count are positive in all four rows. F1 and count have paired CIs
below zero on holdout for both seeds. Some Room or type CIs still cross zero, so
the candidate is promoted to the 1024-case confirmation stage rather than a
final claim.

## Decision

- Crop + parent warm-start + query-set count gate passes two-seed
  train1024/cal256/holdout128 promotion.
- Keep corrected teacher cache as evidence, but not as promoted supervision.
- Do not promote teacher distillation unless Room IoU, instance F1, and count
  improve jointly.
- Do not sweep count thresholds; promotion uses the model-predicted top-k count
  mask and seed-selected query blend only.

## Two-Seed 1024 Promotion

Both seeds improve all four paired metrics on calibration and disjoint holdout:

| Seed / blend | Split | Room IoU delta | F1 delta | Count MAE delta | Type delta |
|---|---|---:|---:|---:|---:|
| seed19 / 0.75 | cal256 | +0.0302 | +0.0768 | -3.20 | +0.1229 |
| seed19 / 0.75 | holdout128 | +0.0300 | +0.0743 | -2.79 | +0.1143 |
| seed23 / 0.25 | cal256 | +0.0579 | +0.0581 | -2.00 | +0.1228 |
| seed23 / 0.25 | holdout128 | +0.0748 | +0.0576 | -2.22 | +0.1060 |

Every listed paired 95% bootstrap interval excludes zero. Across the selected
seed/readout pairs, absolute calibration Room IoU is about `0.203-0.230` and
holdout128 Room IoU is about `0.274-0.319`. This is a promoted room-mechanism
result on the named page-floor protocol, not a complete CAD or external SOTA
claim.
