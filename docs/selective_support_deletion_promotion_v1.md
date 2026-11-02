# Selective Support Deletion Promotion v1

## Mechanism

The frozen dense and query branches produce class probability and query-mask
support. A small deletion head predicts where the footprint should not be
interpreted as room support. Agreed and confident dense/query class regions are
hard-preserved. A topology-safe consumer rejects any deletion that would split
an existing room component and prevents complete deletion of components at least
32 pixels in area.

The class logits are not changed. The operation is therefore one-sided:

```text
blend50 room classes
  -> learned deletion score
  -> hard-preserved identity core
  -> topology-safe keep mask
  -> room-label consumer
```

## Two-Seed Promotion

The promotion rung uses 1024 training and 256 calibration cases, 12 epochs, and
seed-matched 384px dense/query checkpoints on `cuda:0`.

| Split | Seed | Room IoU | Instance F1 | Count MAE | Support IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Holdout128 @ offset256 | 19 | +0.00358 | +0.01140 | -0.35938 | +0.01503 |
| Holdout128 @ offset256 | 20 | +0.00283 | +0.01979 | -0.33594 | +0.01398 |
| Val491 | 19 | +0.00418 | +0.01444 | -0.37882 | +0.01768 |
| Val491 | 20 | +0.00392 | +0.01577 | -0.36049 | +0.01724 |

Every paired 95% interval excludes zero in the favorable direction. The two
seeds therefore pass the primary promotion gate on the disjoint holdout and full
validation. The calibration256 best epochs are also positive on all four
metrics.

## Boundary

- This is predicted-control room identity, not free-text editing.
- The complete wall/opening structural comparison is a separate gate.
- The 4096/512 five-seed scale rung has not yet run.
- The support head changes room support only; it does not fix opening hosts or
  wall vector topology by itself.

## Evidence

- `outputs/reports/selective_support_deletion_promotion_twoseed_v1.json`
- `outputs/reports/selective_support_deletion_seed19_holdout256_v1.json`
- `outputs/reports/selective_support_deletion_seed20_holdout256_v1.json`
- `outputs/reports/selective_support_deletion_seed19_val491_v1.json`
- `outputs/reports/selective_support_deletion_seed20_val491_v1.json`
- `outputs/train/selective_support_deletion_blend50_promotion1024_seed19_v1/report.json`
- `outputs/train/selective_support_deletion_blend50_promotion1024_seed20_v1/report.json`

