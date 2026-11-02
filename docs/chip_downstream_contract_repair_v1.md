# Chip Downstream Contract Repair v1

Date: 2026-10-31

Status: CPU implementation and audit complete. No GPU experiments were run.

## Scope

This repair covers the geometry path after the learning heads:

```text
predicted x0
-> differentiable hard projector
-> constrained exact-area decode
-> refiner trajectory surrogate
-> legalizer-response auxiliary head
-> scale_pack legalization
```

## Fixed Defects

### Autograd-safe projector

The projector previously updated coordinates in-place after deriving their
boxes. Differentiating through the projector produced an autograd version
error. `projector_geometry.py` now uses out-of-place tensor composition for
conditioned projection, boundary snapping and overlap separation.

The endpoint compatibility path now uses the projector as a differentiable
target rather than a detached, no-grad workaround.

### Fixed-shape boundary semantics

Boundary snapping now treats only preplaced blocks as position-immovable.
Fixed-shape blocks keep their shape but may move to satisfy a boundary code.

### Sparse overlap candidates

Blocks are bucketized on a fixed grid and only pairs whose occupied cells
overlap enter the exact pair calculation. The overlap update uses
`index_add_` on per-block shifts rather than maintaining several dense
`N x N` float tensors.

On a four-case, 21-block dev audit, candidate pairs were reduced from 210 to 165
against the full upper triangle.

### Conflict-aware constraint ordering

`conflict_aware` projection evaluates:

1. boundary or MIB first, then overlap;
2. overlap first, then boundary or MIB.

It selects the lower detached joint overlap-boundary-MIB energy per case.
Training and selection masks never read the ground-truth layout.

### Differentiable refiner trajectory

The training path can unroll 2-4 closed-form refiner energy steps with
`create_graph=True`. This reuses Trinity's pinned constraint energy and paper
weights. It does not replace inference-time `refine_closed`.

### Legalizer-response head

A zero-initialized auxiliary head predicts:

- per-block normalized displacement;
- five scale_pack rung logits.

Targets are generated directly from `scale_pack` before/after boxes and
`last_info()["rung"]`. Inference never consumes this head.

### Training projector

The main hierarchical endpoint can now apply the same out-of-place hard
projector during training. Sparse mode, conflict strategy and grid size are
checkpointed model configuration.

## CPU Evidence

- Autograd-safe projector gradient: pass.
- Sparse versus dense overlap update: pass.
- Conflict-aware joint energy backward: pass.
- Two-step differentiable refiner trajectory: finite gradient.
- Legalizer-response displacement and rung gradients: pass.
- Downstream audit:
  `outputs/reports/chip_downstream_contract_audit_v1.json`.
- Structured-head parameter overhead with the response head:
  `54,766` parameters, `0.938%` of the base model.

## Boundaries

These results prove interface, gradient and CPU contract correctness only.
They do not establish better cost, V_rel, b2b, area, congestion, routed
wirelength or timing. GPU quality validation remains blocked.
