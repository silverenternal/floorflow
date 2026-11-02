# Full CubiCasa v1 Results

## Data coverage

- Floor-level cache: `outputs/cache/cubicasa5k_full_v1`
- Samples: 5,126 train / 491 val / 503 test
- Output taxonomy: 65 classes, including door/window subtypes, 24 room groups and 28 furniture groups
- Rare classes are grouped by observed source counts using documented thresholds
- Wall polygons retain thickness; wall direction, opening-wall contact and room adjacency are auxiliary targets
- 50 cases have raster floor images without matching semantic SVG floors; the report marks them as non-trainable raster-only floors

Full data audit:

```text
outputs/cache/cubicasa5k_full_v1/dataset_report.json
outputs/cache/cubicasa5k_full_v1/class_frequency.json
```

## Overfit gate

The 64-floor overfit probe reached teacher-forced accuracy `1.0` at both `t=0.80` and `t=0.95`, showing that the data path, conditioning, 64-channel head and optimizer can fit the training distribution.

```text
outputs/probes/overfit_simplex_fixed/report.json
```

## Main test comparison

Protocol: full 503-floor test split, fixed split order, seed 19, Heun 12 steps, no post-hoc smoothing.

| Metric | Simplex flow baseline | Direct semantic/flow system |
|---|---:|---:|
| Room macro IoU | 0.0635 | **0.0753** |
| Room area coverage | 0.8927 | **0.9866** |
| Opening F1 | 0.00286 | **0.00901** |
| Opening-wall alignment | 0.0888 | **0.9774** |
| Furniture-room containment | 0.7473 | **0.9542** |
| Room area distribution MAE | 0.0470 | **0.0451** |
| Furniture IoU | **0.0787** | 0.0271 |
| Interior-wall IoU | **0.00390** | 0.00235 |
| Outside-footprint fraction | 0.0 | 0.0 |
| Exterior-wall validity | 1.0 | 1.0 |

The direct semantic path substantially improves validity constraints and room coverage. It trades furniture and interior-wall reconstruction IoU against structural stability; this regression is retained in the report.

## Paired bootstrap

Paired bootstrap on the same 503 cases (1,000 resamples):

| Metric | Mean delta | 95% CI |
|---|---:|---:|
| Room macro IoU | +0.01187 | [+0.00645, +0.01699] |
| Opening F1 | +0.00616 | [+0.00279, +0.00968] |
| Opening-wall alignment | +0.88856 | [+0.86484, +0.91209] |
| Furniture-room containment | +0.20695 | [+0.18150, +0.23249] |
| Furniture IoU | -0.05169 | [-0.05713, -0.04677] |
| Interior-wall IoU | -0.00155 | [-0.00238, -0.00070] |

## Publication gate

The design-validity gate passes all requirements on the complete 503-case test split:

- outside footprint = 0
- exterior wall validity >= 0.99
- room coverage >= 0.90
- opening-wall alignment >= 0.85
- furniture-room containment >= 0.90
- room-area distribution MAE <= 0.10

Artifacts:

```text
outputs/eval/full_v1_direct_claim/metrics.json
outputs/eval/full_v1_direct_claim/per_case.json
outputs/eval/full_v1_direct_claim/publication_gate.json
outputs/eval/full_v1_direct_claim/examples/
```

Reference-layout IoU is still low and is treated as a secondary reconstruction metric. Exterior contour to interior design is one-to-many; the publishable claim is constrained design validity, not exact recovery of the reference layout.

## Checkpoints

```text
outputs/train/floorflow_full_v1_b32/best.pt
outputs/train/floorflow_full_v1_semantic/best.pt
outputs/train/floorflow_full_v1_direct/best.pt
```
