# Coarse-to-Fine Hierarchy Feedback v1

Date: 2026-10-30

Status: implemented; rejected at the frozen warm-start quick screen.

## Gap

The current typed hierarchy predicts four state levels:

- L0 global boundary;
- L1 cluster, MIB and boundary tokens;
- L2 per-block position and shape;
- L3 port offsets.

The L0/L1 heads and the L2 backbone are evaluated in parallel. The block
trajectory receives a static typed graph context and a summary of the current
state, but not the model's own predicted boundary or group state. This is a
structural mismatch with coarse-to-fine placement: a block should be predicted
under the boundary and group geometry that the same model has just estimated.

The failed M11 experiment tested a different hypothesis. It added learned group
ordering and Gaussian time phases. Warm-starting from the same production
checkpoint still regressed cost by `0.51%` and V_rel by `11.10%` at the frozen
quick gate, so order phasing is not the missing information path.

## Upgrade

`CoarseToFineCoupler` reorders the existing hierarchy computation without adding
another transformer block.

1. Compute the existing L0 and L1 predictions before the block backbone.
2. Reuse `structural_membership` to pool the predicted group, MIB and boundary
   states for each block.
3. Combine the pooled state with the current block state, typed block token and
   flow time in a hidden/4 bottleneck.
4. Add a zero-gated 3D residual to the L2 trajectory before the existing
   typed-hierarchy backbone.

The module never reads target boxes, target labels, official metrics or
post-route data. It creates no dense `N x N` tensor. The four cluster slots,
typed structural tokens, hierarchy heads and block backbone are all reused.

## Budget

| Model | Parameters | Delta |
|---|---:|---:|
| Base d192 | 5,836,447 | - |
| Coarse-to-fine d192 | 5,858,579 | +22,132 (+0.379%) |

## Invariants

- zero gate preserves the base model prediction exactly;
- block permutation changes the residual by the same permutation;
- direct gradient reaches the bottleneck output;
- checkpoint reconstruction enables the opt-in coupler;
- a one-step real-cache warm-start training smoke is finite and reports a
  nonzero `coarse_to_fine` gradient norm.

## Frozen Screen

- Data: `configs/chip_train_screen_5k_v1.json`;
- seed: 23;
- quick: 2k steps, NFE4/refiner400;
- confirmation: 5k steps, NFE4/refiner1600;
- device: cuda:1;
- gate: no primary regression above `0.2%`, at least one improvement of
  `0.5%`, and legalized feasibility `1.0`.

Passing registers a mechanism candidate only. Full-data, external routed-PPA,
seed replication and venue gates remain mandatory.

## Result

The warm-start arm reached mean cost `-3.78%` but V_rel `+0.53%`. The cost
median worsened and both primary paired intervals crossed zero, so the mean
gain was long-tail driven and the quick gate rejected the mechanism.

## Artifacts

- `src/floorflow/chip_flow/coarse_to_fine_coupler.py`
- `tests/test_chip_coarse_to_fine_coupler.py`
- `configs/chip_coarse_to_fine_warmstart_screen_v1.json`
- `outputs/smoke/chip_coarse_to_fine_v1/coarse_to_fine_warmstart_smoke.json`
- `outputs/reports/chip_coarse_to_fine_warmstart_screen_v1.json`
