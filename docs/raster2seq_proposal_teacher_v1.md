# Raster2Seq Proposal Teacher v1

## Decision

The accepted teacher is a SIGGRAPH 2026 Raster2Seq checkpoint operating on a
privileged, room-class-free structural control. Its room classifications are
explicitly rejected as teacher labels. The cache is used only for
class-agnostic room-mask identity and geometry.

The deployable student will consume the native CubiCasa raster. The teacher
cache and Raster2Seq weights are forbidden at standalone inference.

## Input Screen

All screens used the same val cases and target-only evaluation.

| Input | Cases | Room IoU | Support IoU | Instance F1 | Type accuracy | Decision |
|---|---:|---:|---:|---:|---:|---|
| Raw RGB, unstretched resize/pad | 64 | 0.12642 | 0.63730 | 0.21232 | 0.36467 | reject |
| Structural Canny/adaptive line | 8 | 0.01878 | 0.62803 | 0.02500 | 0.00000 | reject |
| Footprint-cropped raw RGB | 8 | 0.02395 | 0.43392 | 0.00000 | 0.00000 | reject |
| Non-room structural control | 64 | 0.05666 | 0.74998 | 0.43437 | 0.16922 | accept masks only |

The class-specific Room IoU is not a promotion metric for this teacher. The
teacher's class output is unreliable. Support IoU, class-agnostic instance F1,
instance geometric IoU, and count distribution are the usable signals.

## Cache

The cache contains:

- 256 train cases;
- 491 validation cases;
- calibration64 as validation indices 0-63;
- disjoint holdout64 as validation indices 128-191;
- one compressed NPZ per sample with no pickle;
- a manifest with source-raster, cache, taxonomy, contract, checkpoint, and
  repository hashes.

Teacher input removes every FloorFlow room-class pixel. Wall, opening, furniture,
structure, and footprint channels remain. Teacher room masks are predictions,
not target masks.

## Val491 Audit

| Metric | Mean | SD |
|---|---:|---:|
| Room support IoU | 0.76896 | 0.13930 |
| Instance geometric IoU | 0.75023 | 0.20348 |
| Instance F1 at 0.5 | 0.47161 | 0.24038 |
| Room count MAE | 3.6782 | 3.0523 |
| Type accuracy | 0.17435 | 0.28286 |

Calibration64 instance F1 is `0.43437`; disjoint holdout64 is `0.50369`.
Support IoU is `0.74998` and `0.76078`, respectively. The geometry gate passes
on both subsets.

The teacher is not a calibrated count predictor: count MAE is `4.31` on
calibration64. Student cardinality must therefore be learned from supervised
room instances instead of copied from teacher count.

## Interface Contract

The typed cache binds:

- `sample_key = relative_case + "__" + floor_id`;
- deterministic FloorFlow split index;
- FloorFlow taxonomy class ids and names;
- normalized image-frame polygons;
- masks in the same image orientation;
- class probabilities;
- source and cache hashes.

PC-04 must consume this cache through a training adapter, produce an explicit
room plan, and pass a one-batch class/existence/mask/boundary/partition
backward test. Teacher classes and class probabilities are not copied into the
student.

## Evidence

- `outputs/cache/raster2seq_teacher_train256_cal64_v1/train/manifest.json`
- `outputs/cache/raster2seq_teacher_train256_cal64_v1/val/manifest.json`
- `outputs/reports/raster2seq_teacher_cache_audit_v1.json`
- `configs/raster2seq_proposal_cache_contract_v1.json`
