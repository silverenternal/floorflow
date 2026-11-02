# Initial 256px Results

## Protocol

- Dataset: CubiCasa5K test split, 400 cases.
- Input: known exterior footprint only.
- Supervision: 19 foreground channels covering walls, openings, 7 room groups, 8 fixture groups and structure.
- Sampler: Heun, 12 steps, noise scale 0.75, seed 19.
- Decode: raw learned field (`smooth_kernel=1`).
- Boundary: known exterior wall composed after sampling; everything outside the footprint is forced to background.

## Fixed vs learned multi-scale output

| Metric | Fixed 3×3 head | Dynamic 3/5/7 residual head | Delta |
|---|---:|---:|---:|
| Best validation loss | 0.1742 | **0.1691** | -0.0051 |
| Interior pixel accuracy | **0.1521** | 0.1448 | -0.0073 |
| Room macro IoU | **0.0805** | 0.0761 | -0.0044 |
| Room area coverage | **0.7882** | 0.7789 | -0.0093 |
| Interior-wall IoU | 0.0232 | **0.0244** | +0.0012 |
| Door IoU | 0.00444 | **0.00510** | +0.00066 |
| Opening F1 | 0.00614 | **0.00678** | +0.00065 |
| Furniture IoU | 0.0727 | **0.0795** | +0.0068 |
| Closet IoU | 0.0428 | **0.0457** | +0.0028 |
| Appliance IoU | 0.00961 | **0.01065** | +0.0010 |
| Sauna IoU | 0.00710 | **0.01495** | +0.0079 |
| Fireplace IoU | 0.00125 | **0.00419** | +0.0029 |
| Outside-footprint fraction | 0.0 | 0.0 | 0 |

The dynamic head improves the validation objective and most fine-grained object/openings metrics, but does not win every metric: room macro IoU, room coverage and raw pixel accuracy are slightly lower than the fixed head. This is reported as-is rather than cherry-picking the comparison protocol.

## Learned receptive-field routing

The final router report is in:

```text
outputs/train/floorflow_dynamic_head_256/scale_routing_final.json
```

At `t=0.8`, routing is already channel-specific. For example, kitchen averages approximately `0.02 / 0.38 / 0.60` over 3/5/7 kernels, while other-room averages approximately `0.67 / 0.19 / 0.14`. This confirms the head is not a static hand-written per-class rule; the router learns spatial and semantic channel weights during training.

## Artifacts

```text
outputs/train/floorflow_final_256/best.pt
outputs/train/floorflow_dynamic_head_256/best.pt
outputs/eval/floorflow_final_256_test/metrics.json
outputs/eval/floorflow_dynamic_head_256_test/metrics.json
outputs/eval/floorflow_dynamic_head_256_test/examples/
outputs/infer/dynamic_1245/
```

## Main limitation

CubiCasa5K does not annotate loose bedroom/living-room furniture such as beds, sofas or tables. Its stable `FixedFurniture` labels are mostly bathroom, kitchen, closet, sauna and fireplace fixtures. A dedicated bed channel would be untrained on this dataset. Supporting full soft-furnishing design requires an additional dataset or a generated/synthetic annotation layer.
