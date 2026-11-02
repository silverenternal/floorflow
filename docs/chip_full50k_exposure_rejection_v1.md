# Full-50k Training-Exposure Arm Rejection

Date: 2026-10-30

Status: rejected by the frozen same-device advancement gate.

## Protocol

- Cache: `outputs/cache/chip_floorset_50k_train_12k_dev_v2`
- Architecture: typed hierarchy, d192, depth 10
- Training: 50,000 steps, batch 128, gradient accumulation 1
- Nominal exposures: 6.4M
- Evaluation: frozen dev32, cuda:1, NFE4 + refiner400 + projector8
- Reference: `outputs/chip_hierarchy/sota_budget_v1/eval_old_v7_20k_dev32_cuda1.json`

## Result

| Metric | Production 20k reference | Full-50k arm | Relative |
|---|---:|---:|---:|
| Legalized hard cost | 1.189661 | 1.209952 | +1.71% |
| V_rel | 0.042707 | 0.051421 | +20.40% |
| Feasible fraction | 1.0 | 1.0 | unchanged |

The frozen gate requires no primary-metric regression greater than 0.2% and a
meaningful improvement in at least one metric. Both primary metrics regressed, so
the arm fails.

## Decision

- Do not resume this run.
- Do not select an intermediate checkpoint after observing the endpoint.
- Do not automatically extend to a longer training horizon.
- Retain the full curve and final checkpoint as negative evidence.
- Test 21-120 block-count coverage only as an independent frozen-subset branch
  after the current M6 response repair, with its own matched control.

## Artifact

- `outputs/reports/chip_sota_training_full50k_exposure_v1.json`
- `outputs/chip_hierarchy/sota_full50k_exposure_v1/train_50000.log`
- `outputs/chip_hierarchy/sota_full50k_exposure_v1/eval_50000.json`

