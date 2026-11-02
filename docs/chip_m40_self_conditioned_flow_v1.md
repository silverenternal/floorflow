# M40 Self-Conditioned Hierarchy Flow

Date: 2026-10-31

## Mechanism

M27's first NFE call is strong, but later Euler calls were stateless. M40 stores
the previous detached x0 block estimate in `HierarchyState` and injects it
through a zero-initialized low-rank three-dimensional adapter before later
backbone calls.

The first call has no history and is bit-equivalent to M27. Training uses a
fixed 0.5 batch-level keep probability. Inference adds no NFE or refiner step.

## Frozen Screen

- source checkpoint: canonical M27 seed21/data21;
- train subset: 5k cases, 2,000 steps, batch 128;
- control: matched M36 resumed M27 arm;
- evaluation: frozen dev32, NFE4, refiner400, inference seed19;
- feasibility: `1.0` for both arms.

| Metric | Control | M40 | Relative |
|---|---:|---:|---:|
| Cost | 1.13870 | 1.23729 | +8.66% |
| V_rel | 0.03651 | 0.04291 | +17.54% |
| Area gap | 0.07278 | 0.13040 | +79.18% |
| b2b HPWL gap | 0.02477 | 0.05435 | +119.43% |
| p2b HPWL gap | 0.00857 | 0.45725 | +52.34x |

The cost, area and b2b paired bootstrap intervals exclude zero in the harmful
direction.

## Why the Training Curve Was Misleading

M40 improves latent MSE, rho MAE and area on the 500-case training curve, and
also improves raw V_rel by `0.00966` absolute and raw boundary by `0.46875`.
Raw HPWL changes by only `+0.00027`.

The legalized endpoint still collapses. Two high-complexity cases dominate:

- case 11822, 114 blocks: legalizer rung `3 -> 5`, solves `7 -> 19`;
- case 409, 85 blocks: legalizer rung `3 -> 5`, solves `11 -> 19`.

Thus the failure is not a uniformly worse raw vector field. The generated raw
layout is harder for the existing legalizer to pack at high block counts. A
future architecture residual must report legalizer rung and solve counts as
primary mechanism diagnostics, not only HPWL/V_rel.

## Decision

Reject M40. Do not sweep the keep probability or adapter rank. Pause the
trajectory-residual family until a new mechanism explicitly targets legalizer
stability.

## Artifacts

- `configs/chip_m40_self_conditioned_flow_v1.json`
- `outputs/reports/chip_m40_self_conditioned_quick_refiner400_dev32.json`
- `src/floorflow/chip_flow/hierarchy_state.py`
- `src/floorflow/chip_flow/hierarchical_flow_runtime.py`
- `src/floorflow/chip_flow/train_objective.py`
- `tests/test_chip_self_conditioned_flow.py`
