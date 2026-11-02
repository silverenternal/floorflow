# Chip Optimizer Contract Repair v1

Date: 2026-10-31

Status: CPU implementation and audit complete. No GPU experiments were run.

## Scope

This repair covers the layer after the training objective:

```text
loss
-> backward
-> finite guards
-> semantic gradient clipping
-> AdamW parameter groups
-> EMA
-> final/best checkpoint selection
```

## Implemented Changes

### Semantic parameter groups

The optimizer now separates:

- backbone decay;
- backbone no-decay;
- core model decay;
- core no-decay;
- structured-head decay;
- structured-head no-decay;
- legalizer-response decay;
- legalizer-response no-decay.

Bias, normalization scale, residual scale and gate parameters use zero weight
decay. Backbone, structured and legalizer groups have independent LR scales and
head warmup lengths.

### Gradient and numeric safety

- Every micro-batch loss and reported loss term must be finite before backward.
- A single global finite-gradient check runs before clipping.
- Each semantic optimizer group is clipped independently.
- Diagnostics record pre-clip norm, post-clip norm and clip coefficient.
- Non-finite parameters after `optimizer.step()` fail closed.

### BF16/FP32 boundary

The normal model forward remains inside the existing BF16 autocast path.
Projector displacement, refiner energy and differentiable refiner trajectory
are executed with autocast disabled and explicit FP32 tensors.

### EMA

Two policies are supported:

- `fixed`: exact existing `0.9995` control;
- `warmup`: decay ramps linearly from zero to the configured base decay over a
  declared number of optimizer updates.

The selected decay is recorded in step diagnostics.

### Checkpoint policy

`checkpoint_policy` is frozen before training:

- `final`: use the last-step state;
- `best`: use the best validation checkpoint.

The final report records the selected step, selected checkpoint and validation
state. Existing runs retain the `final` policy by default.

## Reused Project Code

- `floorflow.training.gradient_health.nonfinite_gradient_names`
- `WarmupCosineScheduler` scheduling conventions
- CFM grouped-optimizer and ModelEMA design
- existing checkpoint/EMA/state serialization contracts

## CPU Audit

Script:
`scripts/audit_chip_optimizer_contracts.py`

Report:
`outputs/reports/chip_optimizer_contract_audit_v1.json`

The audit verifies:

- all eight parameter groups;
- no-decay bias/norm/gate routing;
- independent head LR and warmup;
- finite fail-closed grouped clipping;
- fixed/warmup EMA boundary values;
- finite CPU gradient flow through the full structured model.

## Boundary

No quality improvement is claimed. GPU convergence, seed stability and
legalized metrics remain pending.
