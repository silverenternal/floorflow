# Chip Training-Frontier 20k Rejection v1

Date: 2026-10-30

## Decision

Reject the S1 20k combined training recipe. Do not attribute its failure to
training exposure alone and do not resume its interrupted 50k continuation.

The failed arm combined four changes at once:

- frozen 20k-case subset versus the 50k training cache;
- effective batch 256 versus 128;
- projector iterations 8 versus 4;
- a four-parameter `type_bias` tensor present in the current code but absent
  from the older production checkpoint. The tensor is unused in `typed` mode.

## Same-Device Comparison

Both checkpoints were evaluated on `cuda:1`, frozen dev32, NFE4 with the
400-step refiner, project iteration count 8 and identical legalization.

| Arm | Cost | V_rel | Feasibility |
|---|---:|---:|---:|
| Old production 20k | 1.189661 | 0.042707 | 1.0 |
| New combined 20k | 1.260335 | 0.053877 | 1.0 |
| Relative change | +5.94% | +26.15% | 0 |

Paired case statistics:

- cost mean delta `+0.070674`; 10 improved, 21 worsened, 1 unchanged;
- V_rel mean delta `+0.011170`; 7 improved, 14 worsened, 11 unchanged;
- the 95% percentile-bootstrap intervals both cross zero because of case
  variance, but the mean gates and case direction fail decisively.

The failed arm also regressed b2b HPWL gap by `+38.11%` and rho/aspect MAE by
`+13.10%` against the production control.

## Evidence

- `outputs/chip_hierarchy/sota_budget_v1/eval_old_v7_20k_dev32_cuda1.json`
- `outputs/chip_hierarchy/sota_budget_v1/eval_new20k_dev32_cuda1.json`
- `outputs/reports/chip_sota_training_old_vs_new20k_dev32_cuda1.json`
- `outputs/reports/chip_sota_training_compute_ladder_v1.json`

The interrupted 50k arm reached logged step 3050. Its unreferenced rolling
checkpoint was removed, while the log and terminal record remain.

## Next Controlled Test

`configs/chip_sota_training_subset_control_v2.json` restores the production
training settings (20k subset, batch128, gradient accumulation 1, projector4)
for 20k optimizer steps. It isolates the frozen data subset before any longer
horizon or batch-size change.

If that control still fails, the current 20k subset/order is not suitable for a
training-exposure frontier. If it passes, effective batch and projector
settings are the next isolated variables.

## Clean Subset Control Outcome

The clean 20k-subset control restored batch128, gradient accumulation 1 and
projector4. It still failed on `cuda:1`, frozen dev32, NFE4/refiner400:

| Arm | Cost | V_rel | Feasibility |
|---|---:|---:|---:|
| Old production full-50k 20k | 1.189661 | 0.042707 | 1.0 |
| Clean 20k-subset control | 1.240490 | 0.056765 | 1.0 |
| Relative change | +4.27% | +32.92% | 0 |

The paired cost interval `[+1.17%, +9.06%]` and V_rel absolute interval
`[+0.21%, +2.65%]` both exclude zero. The current frozen 20k subset is therefore
not a valid proxy for the training-exposure frontier. Further batch/projector
isolation on this subset is rejected; the next allowed test is one training
exposure arm on the full 50k cache after U1 screening releases the GPU.

## Boundary

This is a dev32 training diagnostic, not an Official-100 result, not a
matched-Trinity result and not evidence against every longer-training recipe.
