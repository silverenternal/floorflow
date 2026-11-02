# Order-Conditioned Group Staging v1

Date: 2026-10-30

Status: implemented; rejected at the frozen warm-start quick screen.

## Motivation

The current typed hierarchy carries:

- L0 global boundary state;
- four typed cluster slots from FloorSet group labels;
- L2 per-block positions and shapes;
- L3 port state;
- an invariant typed netlist context reused across the flow.

All blocks currently follow simultaneous trajectories. U1's dynamic edge relay
and M10's projection-consistent trajectory both failed confirmation. Recent
primary sources identify a different untested factor:

- OrderPlace (`arXiv:2606.08904`) treats macro placement order as a first-class
  optimization variable;
- Re2MaP (`arXiv:2511.08054`) recursively prototypes/relocates macro groups and
  defers blocks to later rounds.

## Minimal Extension

`OrderedGroupStaging` reuses the four existing cluster tokens and adds one
zero-gated residual per block.

1. Project each cluster token to a learned case-specific priority.
2. Convert priorities to differentiable normalized phase centers.
3. Evaluate a Gaussian time-phase gate for each group at the current flow time.
4. Gather the cluster token, phase token and current block state for each block.
5. Add a bottleneck residual to the existing backbone input.

The module never accesses target boxes, target classes, official metrics or
post-route information. Group 0 and unseen groups remain unchanged.

## Parameter and Compute Budget

| Model | Parameters | Delta |
|---|---:|---:|
| Base d192 | 5,836,447 | - |
| Order-conditioned d192 | 5,884,500 | +48,053 (+0.823%) |

The residual uses a hidden/4 bottleneck. It does not create dense NxN attention
or another transformer block.

## Invariants

- zero-gate output is exactly equivalent to the base model;
- direct gradient reaches the residual and gate;
- block permutation changes output only by the same permutation;
- checkpoint loading reconstructs the opt-in module;
- a real-cache CPU training smoke completes with finite loss.

## Frozen Screen

- Data: `configs/chip_train_screen_5k_v1.json`;
- Seed: 23;
- Quick: 2k steps, NFE4/refiner400;
- Confirmation: 5k steps, NFE4/refiner1600;
- Device: cuda:1 because an unrelated external DCDGAN job owns cuda:0;
- Eligibility: no primary regression greater than 0.2%, at least one improvement
  of 0.5%, and legalized feasibility 1.0.

Passing the frozen screen registers a mechanism candidate only. Full-data and
external routed-PPA gates remain mandatory.

## Scratch-Training Negative

The first frozen 5k screen trained both arms from scratch. It failed the quick
gate: cost `+4.57%` and V_rel `+1.72%` with feasibility 1.0. The learned global
gate was `-0.0505`; active group phase centers clustered around the middle of
the trajectory. This arm is retained as `rejected_quick_screen`.

The repair variant warm-starts both arms from the same production 20k
checkpoint and adapts for the same frozen budget. Candidate-only new
parameters are admitted explicitly, and the frozen screen is rerun under a new
config/report path.

The warm-start repair also failed: cost `+0.51%` and V_rel `+11.10%` versus
the matched control. Group-time phasing is retained as a negative result.

## Artifacts

- `src/floorflow/chip_flow/ordered_group_staging.py`
- `tests/test_chip_ordered_group_staging.py`
- `configs/chip_order_staged_screen_v1.json`
- `outputs/reports/chip_order_staged_screen_v1.json`
- `outputs/reports/chip_order_staged_warmstart_screen_v1.json`

