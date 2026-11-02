# Chip Structured Head Repair v1

Date: 2026-10-31

Status: CPU implementation and contract tests complete. No GPU training or
evaluation was run.

## Motivation

The previous chip hierarchy learned four endpoints but had five structural
weaknesses:

1. L0/L1 predictions did not condition the same-call L2 trajectory.
2. The block x0 target was effectively supervised again after projection.
3. Center and rho/aspect shared one L2 output head.
4. Connected and unconnected pins shared one port semantic path.
5. Training did not optimize for refiner or legalizer compatibility.

## Implementation

### Same-call hierarchy routing

`MembershipHierarchyCoupler` consumes the predicted L0 bbox and predicted L1
cluster/MIB/boundary states. It routes predicted structural states and tokens to
each block through the existing membership masks. Separate zero-gated global and
structural paths preserve exact base-model behavior when disabled.

### Block supervision

`head_supervision.py` separates:

- free center coordinates;
- free rho/aspect;
- fixed-shape rho;
- preplaced center and rho.

The structured contract masks dimensions already fixed by conditioned
projection. It also suppresses the duplicate post-projection block decode loss
that previously repeated the main block x0 objective.

### Factorized L2 output

`FactorizedBlockHead` adds independent low-rank center and rho/aspect residuals
over the existing shared Trinity backbone. It does not create a second
backbone.

### Port semantics

`DualSemanticPortHead` separates connected-pin offset residuals from
unconnected-pin alignment residuals. Structured supervision normalizes the two
groups separately instead of letting the larger group dominate the mean.

### Refiner and legalizer compatibility

`endpoint_compatibility_terms` is training-only. It evaluates:

- the existing hard-projector displacement;
- physical energy before and after projection;
- the pinned Trinity closed-form refiner energy.

Inference keeps the established NFE, refiner, projector and legalizer path.

### Output provenance

Typed hierarchy configurations now record `output_mode="x0"`, matching the
effective backbone head.

## CPU Evidence

- Structured-head unit tests: pass.
- Head-supervision tests: pass.
- Endpoint-compatibility tests: pass.
- Existing hierarchy/adapter tests: 71 passed.
- Parameter overhead: `42,807`, or `0.733%` over the 5,836,447-parameter base.
- CPU audit: `outputs/reports/chip_head_contract_audit_v1.json`.

The audit covers 50,000 training cases:

- fixed blocks per case: `3.960`;
- preplaced blocks per case: `1.549`;
- anchored blocks per case: `5.355`;
- connected-pin fraction: `66.54%`;
- cases with every pin connected: `11.46%`.

## Boundaries

This implementation proves interface compatibility, finite gradients,
permutation behavior, mask routing and checkpoint restoration. It does not prove
quality improvement. The first GPU action remains a matched
2k/refiner400 quick screen against the untouched base, followed by
5k/refiner1600 confirmation only if the quick gate passes.
