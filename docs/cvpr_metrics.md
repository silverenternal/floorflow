# CVPR metric record

## Claim boundary

The accepted result is **layout-conditioned CAD refinement**:

```text
contour
+ explicit room program
+ coarse room / wall-opening / furniture layout control
-> layered 256x256 CAD output
```

It is not a contour-only exact interior reconstruction claim. Contour-only
metrics remain a diagnostic and are not mixed into the main result.

## Frozen CubiCasa 503 gate

All values below are single deterministic outputs, with no best-of-N, human
rating, or per-case test selection.

| Metric | Value | Threshold |
|---|---:|---:|
| Room macro IoU | 0.9718 | 0.95 |
| Instance geometric IoU | 0.9711 | 0.95 |
| Instance room F1@0.5 | 0.9740 | 0.95 |
| Instance type accuracy | 1.0000 | 0.95 |
| fpeval wall F1@0.05 | 0.9637 | 0.95 |
| fpeval opening F1@0.05 | 0.9724 | 0.95 |
| fpeval polygon-room F1 | 0.9818 | 0.95 |
| Furniture IoU | 0.9672 | 0.95 |
| Furniture type accuracy | 0.9774 | diagnostic |
| Room-support IoU | 0.9741 | 0.95 |
| fpeval validity | 1.0000 | 0.95 |
| Conditional watertightness | 0.9595 | 0.95 |

Unconditional watertightness is `0.8986`; the GT itself is only `0.8827`
watertight under the same verifier. Both values are reported without hiding.
The conditional metric uses only target-watertight samples.

Evidence:

- `outputs/eval/layered_pipeline_test503_frozen_v3_polygon_room/metrics.json`
- `outputs/reports/b2_layered_pipeline_gate_test503.json`
- `outputs/reports/a2_layered_room_gate.json`
- `outputs/reports/a1_mita_ablation.json`

## Cross-dataset skv4 296

| Metric | Value |
|---|---:|
| Room macro IoU, 284 labeled plans | 0.9857 |
| Instance geometric IoU | 0.9685 |
| Instance room F1 | 0.9713 |
| Instance type accuracy | 0.9899 |
| fpeval wall F1 | 0.9799 |
| fpeval opening F1 | 0.9648 |

Twelve of 296 plans have no GT room polygons; room and instance metrics are
undefined for them and are excluded only from those averages.
Evidence: `outputs/reports/b3_skv4_gate_test296.json`.

## Ablations and controls

- MiTA vs MHA used identical initialization, a 1,024-sample train prefix,
  3 epochs, and 191 validation cases. MiTA improved room, instance, F1, and
  type metrics; routing entropy was `0.970`, effective-expert fraction `0.682`.
- Room-program hard gating removed systematic false `Alcove` instances and
  raised instance F1 from `0.8615` to `0.9740`.
- Separate layers prevent wall/opening/furniture from overwriting room storage.
- Room support is supervised by polygon support, not approximated by the
  visible room raster.

## Remaining research work

1. Replace target-derived coarse controls with a contour-to-control predictor.
   Current target-free CFM/control baselines remain at room IoU
   `0.0607/0.1630/0.7196`.
2. Improve contour-only condition identifiability; train-only contour-neighbour
   analysis finds only `0.596%` of test plans with a `>=0.95` neighbour.
3. Promote distributional generation only after its sample quality can support
   the same `0.95` gate as the deterministic refinement path.
