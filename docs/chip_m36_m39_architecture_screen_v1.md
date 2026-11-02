# Chip Architecture Screen M36-M39

Date: 2026-10-31

## Purpose

The canonical M27 checkpoint is retained as the source model. Every screen in
this cycle warm-starts from that exact checkpoint and uses the frozen 5k-case
training subset, seed 21, NFE4, refiner400, and the frozen 32-case legalized
development manifest. No exploratory arm uses full training data.

The matched control is M27 resumed for 2,000 steps without an architecture
change. It is shared by all four architecture arms.

## Evaluated Architectures

### M36: hidden-token hierarchy bridge

The bridge cross-attends predicted L0/L1 states into the Trinity hidden tokens
before all twelve backbone layers. It adds 189,030 parameters and is exactly
equivalent to M27 at initialization.

The legalized endpoint regresses cost by 3.75% and V_rel by 1.09%. Grouping
improves 9.09%, but b2b and p2b regress 66.47% and 34.07x respectively. The
mechanism is too invasive for local geometry.

### M37: bounded structural bridge

This variant removes current block coordinates from the bridge query, uses a
zero gate, and bounds the residual with `tanh`. Its training curve improves,
but the legalized endpoint regresses cost 5.24% and V_rel 21.00%. The learned
gate is effectively zero (`-3.54e-5`). Retain the training/legalized optimism
as an evaluation warning.

### M38: global bbox consensus

This output operator aligns the predicted block envelope to the model's L0
forecast while preserving relative block displacement and rho. It adds one
parameter and is exactly equivalent at initialization.

Mean V_rel improves 3.04% and grouping improves 18.18%, but cost regresses
2.90%, area gap regresses 45.32%, and b2b regresses 67.43%. The learned gate is
small (`0.03295`) yet the effect is not structurally safe.

### M39: structural-block-only bridge

This variant applies the M36 hidden-token bridge only to cluster, MIB, or
boundary-coded blocks. Cost regresses 4.84% and V_rel regresses 25.30%; the
V_rel paired interval excludes zero. Restricting the bridge does not preserve
its grouping signal.

## Decision

Reject the entire M36-M39 bridge/consensus family. The shared failure is that
coarse hierarchy information injected into an already pretrained vector field
changes local geometry before its structural benefit is realized.

Do not tune bridge width, gate values, or mask thresholds against the frozen
development set. The next architecture candidate must satisfy all three
conditions:

1. exact M27 output at the first denoising call;
2. access only information produced by earlier calls in the same trajectory;
3. no additional NFE or refiner steps.

Self-conditioning on the previous x0 estimate is the next admissible design.
It should be trained with a fixed drop probability and compared with the same
M36 control before any confirmation run.

## Artifacts

- `outputs/reports/chip_m36_m39_architecture_screen_v1.json`
- `configs/chip_m36_hierarchy_token_bridge_v1.json`
- `configs/chip_m38_global_bbox_consensus_v1.json`
- `src/floorflow/chip_flow/hierarchy_token_bridge.py`
- `src/floorflow/chip_flow/global_bbox_consensus.py`
