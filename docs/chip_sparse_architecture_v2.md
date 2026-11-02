# Chip Sparse Architecture v2

Status date: 2026-10-26. This note records the architecture-only completion
pass for route B. No model was trained and the old sparse checkpoint is not
reused as evidence for the revised architecture.

## Planning Gap

The first sparse implementation had the right top-level decomposition but four
production gaps:

1. The token decoder still used dense self-attention after sparse slot readout.
2. The graph encoder converted b2b edges in one direction only and did not
   accept the native constraint tensor explicitly.
3. Decoder budgets were forced to `active_slots * decoder_top_k`, so router
   allocation could not vary within a case.
4. Parameter matching and route diagnostics were implemented only in ad-hoc
   experiment scripts.

## Revised Data Path

```text
features + constraints [B,N,5]
    -> b2b reverse-edge messages + pin-to-block messages
    -> graph-conditioned block tokens
    -> learned slot queries + per-case slot budget
    -> top-k token-to-slot assignments
    -> OfficialMiTARouteAttention over active parent slots
    -> residual token MLP
    -> exact-area block decode
```

`SparseHierarchyEncoder` owns the graph and parent-slot encoder. The same
encoder is instantiated for condition and target feature widths; no second dense
global encoder is used in the sparse arm. `decoder_router_budgets` passes slot
weights, active-slot capacities, and case complexity to the existing
`allocate_group_token_budgets`.

## Reuse

| Need | Existing implementation |
|---|---|
| Sparse landmark routing | `floorflow.model.official_mita_route.OfficialMiTARouteAttention` |
| Dynamic integer budgets | `floorflow.model.dynamic_budget.allocate_group_token_budgets` |
| Exact-area latent decode | `floorflow.chip_flow.model.decode_boxes_batch` |
| Constraint physics | `floorflow.chip_flow.batched.batch_physics_terms` |
| Cached FloorSet substrate | `floorflow.chip_flow.cache.PackedCaseStore` |

## No-Training Audit

Run on the official 100-case cache:

```bash
uv run python scripts/audit_chip_architecture.py \
  --batch-size 16 \
  --sampling-steps 1
```

The audit checks native graph/constraint inputs, learned parent slots, sparse
MiTA readout, the absence of dense sparse-arm attention, variable active slots
and router budgets, parameter matching within tolerance, and the 600-line source
limit. It reports parameters, wall-clock, top-k, budget, slot, and router-load
diagnostics for every arm. The generated report is
`outputs/reports/chip_architecture_audit_v2.json`.

Expected default configuration:

| Arm | Parameters |
|---|---:|
| flat token | 5,415,939 |
| global slot (192) | 5,493,891 |
| production sparse | 4,388,358 |
| matched global slot (168/208) | 4,354,939 |

The matched control differs by 0.76%, inside the audit's 2% tolerance. These are
architecture counts only; they are not a quality result.

## Checkpoint Boundary

The parameter shape changed in this pass. `outputs/chip_sparse/v1` and all older
sparse checkpoints must not be loaded or reported as v2 evidence. Resuming the
route-B experiment requires a fresh two-step wiring smoke, then the
preregistered matched training run.
