# Chip Architecture Completion v2

Status date: 2026-10-31. This audit covers CPU implementation and contract
evidence only. No GPU quality evaluation, long training or ORFS run is part of
the evidence package.

A short GPU readiness probe and bounded smoke were accidentally started before
the no-GPU constraint was applied. No active GPU job remains. Those artifacts
are retained only for audit transparency and are explicitly invalid for
mechanism confirmation, promotion or claims.

## Canonical lineage

The canonical implementation lineage is
`typed_hierarchy_m27_trinity_backbone_init`. The older parameter-matched
graph-token and global-slot studies remain provenance, not interchangeable
evidence.

The canonical M27 checkpoint has 6,797,217 parameters. This is 16.06% above
Trinity d192 (5,856,400). Its 5,000 steps at micro-batch 128 expose 640,000
samples, versus Trinity's nominal 51.2M, while starting from the Trinity d192
backbone. The audit is machine-readable in
`outputs/reports/chip_architecture_lineage_audit_v1.json`.

The complete v2 switch set adds 90,300 parameters, or 1.33% over M27, for a
total of 6,887,517 parameters (17.61% above d192).

## Implemented architecture repairs

### Same-call hierarchy and coarse commitment

- `MembershipHierarchyCoupler` routes predicted L0/L1 states into L2 in the same
  forward call.
- `coarse_commit_fraction` freezes L0/L1 after a predeclared trajectory
  fraction, while L2/L3 continue evolving.
- NFE1 uses threshold 1 so the first and only prediction step is not frozen.
- The disabled path remains exactly equivalent to the parent model.

### Grouping-aligned relation state

- `structural_relations.py` predicts cluster geometry, gap, and connected-
  component deficit.
- Target connectivity is computed from GT boxes only during training.
- The relation token conditioning enters the existing L1 token path rather than
  creating a second hierarchy stack.
- Explicit block count, b2b/p2b/pin density, area variation, and edge-weight
  statistics are injected into the global token through a zero-initialized
  complexity condition.

### Relation-specific graph

- Structural queries receive only their cluster, MIB, or boundary relation.
- Constraint queries receive only anchored, boundary-coded, or cluster/MIB
  members.
- Port keys remain tied to p2b incidence.
- Four zero-initialized relation gates are independent.
- A relation-gate regularization penalizes both inactive and collapsed gate use.
- `dynamic_graph_input` reuses `StateEdgeRelay` to inject current b2b/p2b
  geometry before the backbone instead of adding a post-output residual.

### Factorized geometry and legalization loop

- `FactorizedBlockHead` keeps separate center and rho/aspect residual fields
  over one shared backbone.
- Existing projector, differentiable refiner trajectory, endpoint compatibility
  and legalizer-response heads are enabled through the v2 recipe.
- The predicted legalizer displacement can act only at the final sampling step,
  with a configurable per-block trust radius; internal ODE steps do not apply it.
- A CPU-only 32-case sidecar now converts shelf-source layouts through exact
  scale-pack, supplying displacement/rung labels for the first closed-loop
  screen without requiring GPU generation.
- Legalizer-response labels reuse the existing scale-pack sidecar.

### Routeability field

The existing `SpatialOccupancyCoupler` now optionally rasterizes:

- block extent samples rather than center-only points;
- b2b midpoint demand;
- p2b pin-to-block demand.

The original three-channel occupancy path remains available for reproduction.

### Source and training stability

- Mixed source training can anneal the shelf probability toward deployment.
- Backbone staged unfreeze keeps frozen parameters in optimizer groups and
  enables them at a configured step.
- The evaluator stays frozen across source changes.

### Frozen execution plan

`configs/chip_v2_execution_plan_v1.json` freezes the order
`M50 -> M51 -> M53 -> M52 -> M55`, the shared 5k-case training subset, the
32-case legalized screen, matched candidate/control budgets, and the exact
checkpoint/RNG contract. M54 is a legalizer-response side stage opened only
after M53.

The CPU audit verifies that M53 does not apply the legalizer action early, M54
requires response labels before action, M55 enables the route field and source
schedule, and GPU execution remains `not_authorized`. The checkpoint test also
round-trips Torch, NumPy, Python, batch and balanced-sampler state.

### Portfolio

- `balanced` selection protects b2b HPWL, p2b HPWL, outline compactness and
  aspect after the V_rel tolerance.
- `PortfolioRouter` and `evaluate_chip_router_portfolio.py` implement one
  routed trajectory per case under a baseline budget.
- CPU fitting from the three observed M48 reports reaches about 63% train and
  only 48% case-holdout three-way profile accuracy after adding area,
  edge-weight and pin-spread features. The router overfits the observed rows
  and remains development-only.

## Reuse decisions

- Trinity backbone, spectral PE, closed refiner, constraint energy and
  scale-pack remain the execution core.
- Existing typed graph, hierarchy adapters, spatial coupler, end-point
  compatibility, legalizer-response head, selector and checkpoint IO are
  extended rather than duplicated.
- MiTA remains available through the project's adapted route-attention module.
  Its upstream module requires FlashAttention 2.6.3 and a different token/mask
  contract, so it is not inserted into the small typed cross-attention path
  without a matched shape and latency experiment.

## CPU evidence

- `outputs/reports/chip_architecture_v2_contract_audit.json`
- `outputs/reports/chip_v2_execution_plan_audit_v1.json`
- `outputs/reports/chip_current_evidence_gate_postfix_v1.json`
- `outputs/reports/chip_architecture_salvage_matrix_v1.json`
- `outputs/reports/chip_nfe1_hierarchy_causality_v1.json`
- `outputs/reports/chip_complexity_tail_audit_v1.json`
- `outputs/reports/chip_route_coordinate_audit_v1.json`
- `outputs/reports/chip_portfolio_router_fit_v1.json`
- `tests/test_chip_architecture_v2.py`
- Existing projector, structured-head, training-runtime and data-pipeline tests.

The v2 zero-gate path reproduces the parent model exactly on the contract batch.
All newly added paths have CPU gradient or sensitivity coverage.
Full CPU verification is `1353 passed`; scoped Ruff passes and every first-party
Python file remains below 600 lines.

The accidental cuda:0 readiness and mechanism-smoke artifacts are excluded from
the evidence chain. They are not treated as profile results, quality results or
architecture confirmations.

The existing post-fix portfolio tail audit shows that the remaining weakness is
not uniformly distributed: dense-pin cases above 9 p2b edges per block have
essentially neutral cost and a 36% V_rel gain but about 5.7x worse b2b HPWL than
the anchor. Cases above 10 b2b edges per block retain a 47% b2b gap. This is the
primary reason the v2 relation and complexity paths remain mandatory rather
than optional capacity.

The lineage audit reports no matched-budget anchor: the portfolio uses NFE2,
3200 refiner steps and three legalizer calls, while every available Trinity
anchor uses NFE1, 1600 refiner steps and one legalizer call. This remains an
absolute-budget diagnostic until an equal-budget anchor exists.

The route-coordinate audit deliberately fails review: pins are physical
coordinates while block latents are normalized, and the old scale-only pin
normalization produced no all-pin-inside-bbox cases. The route field now uses
valid-pin min/max with a chart-scale fallback, but the remaining block-center
chart alignment must be resolved before claiming routeability transfer.

## Open GPU gates

The following are deferred and require later explicit authorization:

1. M50/M51 independent quick screens on a frozen subset.
2. M52 relation and dynamic-graph matched ablation.
3. M53/M54 legalizer-compatible endpoint confirmation.
4. M55 route-field sensitivity on external designs.
5. Equal-budget Trinity anchor and three-training-seed finalist confirmation.

No result in this document supports a SOTA or routed-PPA claim.
