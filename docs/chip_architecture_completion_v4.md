# Chip Architecture Completion v4

Status date: 2026-11-01. This package is an implementation and CPU-contract
audit. It launches no GPU experiment, no long training run and no ORFS job.
Nothing here is evidence of improved cost, V_rel, routed PPA, congestion or
timing.

## Ordering

The repair order follows the model data path:

1. explicit target-free global design specification;
2. semantic node and port features;
3. dynamic netlist hyperedges inside the flow trajectory;
4. per-cluster hierarchy and legalizer-response residual paths;
5. analytical routed-wirelength reranking;
6. candidate consensus as an optional distillation target;
7. checkpoint, resume, configuration and engineering verification.

The canonical opt-in configuration is
`configs/chip_architecture_completion_v4.json`.

## Architecture branches

### Global specification

`global_spec.py` constructs seven normalized features from core size, die size and
macro area: normalized core width/height, core aspect, die-to-core width/height
ratios, macro utilization and core-to-macro area ratio. `GlobalSpecConditioner`
has a zero-initialized output and gate, so loading its parameters into a parent
model is numerically inert.

External ORFS/DEF graphs derive this vector from design geometry and block area,
not target positions. Image-floorplan batches do not provide a core/die spec.
Enabling the branch without `global_spec` therefore raises an error instead of
silently training a dormant module.

### Node and port semantics

The legacy 19-column block features remain unchanged. Separate optional tensors
carry:

- macro orientation, fixed/unplaced status and normalized shape;
- BTerm side/direction/layer semantics.

The encoders reuse the existing masked zero-gated residual pattern. Branch-on
without its feature tensor is a contract error. DEF and ORFS orientation forms
are canonicalized through one alias table (`R0`, `R90`, `R180`, `R270`, `MY`,
`MX`, `MY90`, `MX90` and the eight canonical names).

Port features, model pins, p2b indices and `net_port_incidence` use one sorted
set of ports participating in macro-bearing nets. Unused design ports are omitted
consistently instead of shifting the semantic rows.

### Dynamic netlist hyperedges

`netlist_hypergraph.py` reads an optional net incidence contract derived from the
existing external macro graph. At each flow step it pools current block center,
span, state and learned token over valid multi-terminal nets, then scatters the
net message back to member blocks. It is zero-gated and permutation-equivariant.

`net_features` and `net_block_incidence` must arrive together. The encoded feature
width is fixed at four, and invalid nets are removed before both pooling and
degree normalization. The conditioner projects its per-block message directly
to the three-dimensional flow-state residual; it does not add a hidden-width
token tensor to geometry space.

### Hierarchy and legalizer endpoints

The standalone membership branch reuses `MembershipHierarchyCoupler` and routes
predicted L0/L1 state only to member blocks. The standalone legalizer branch
reuses `LegalizerResponseHead` from the structured suite.

The trust-region projection helper is shared by structured and standalone
paths, so there is no second legalizer-response implementation. Only the final
sampling step can apply the action. Hyperedge-on without a netlist context is a
contract error.

### Analytical external reranking and consensus

`select_candidate_with_analytical_response` first applies the existing legality,
route and solver-work guards. It rejects missing/non-finite analytical scores and
uses routed-wirelength before congestion and displacement tie-breaks.

`CandidateConsensusFuser` consumes candidate states and target-free risk
features. It is permutation-equivariant over candidates, requires the declared
reference candidate to be valid, and starts from that reference through a
zero-gated residual. This is a distillation representation only; it is not a
deployed portfolio claim.

Masked conditioners and invalid padded candidates are sanitized before network
evaluation, so a masked `NaN` cannot survive through `0 * NaN`. A finite value
that is marked valid still fail-closes. Analytical reranking also rejects
duplicate names, non-positive/nonfinite routed scores and invalid congestion.

## Training controls

`hierarchy_source_max_shelf_probability` caps the scheduled mixed-source
probability in `[0, 1]`. The source and global-spec controls are serialized in
the training configuration. Resume validation rejects changes to all new
architecture flags when optimizer state is restored.

`load_chip_checkpoint` reconstructs the global-spec, semantic, hyperedge,
membership and standalone-legalizer branches from config and permits missing
parameters only when the checkpoint genuinely predates that opt-in branch.

## Reuse and engineering constraints

- The typed encoder and relation masks are extended rather than duplicated.
- Membership and legalizer-response heads reuse structured-head implementations.
- External net membership reuses DEF/ORFS graph parsing.
- Legalization and refinement continue to call the pinned Trinity modules.
- Every first-party Python file must remain strictly below 600 lines.

The central files were already close to the limit. Hierarchy, hyperedge and
legalizer branches were extracted into `chip_hierarchy_conditioning.py` so
`hierarchical_flow.py` remains below 600 lines.

## Verification

Focused contract coverage lives in
`tests/test_chip_architecture_completion_v4.py`. It checks:

- zero-gate identity and explicit-feature sensitivity;
- missing-input fail-closed behavior;
- hyperedge state dependence and gradients;
- hyperedge residual width and full external-graph-to-model execution;
- trust-radius clipping;
- analytical selection guards;
- candidate identity prior, validity checks and permutation equivariance;
- DEF/ORFS orientation aliases;
- checkpoint reconstruction and resume mismatch rejection.

The final hardening audit additionally checks unused-port alignment, masked
nonfinite rows, padded-candidate isolation and production-context failures
without relying on Python `assert`.

The final CPU gate is full `pytest`, scoped Ruff, JSON validation and a repository
line-length audit. Passing that gate establishes implementation readiness only.
The next scientific step is a later matched-subset experiment with a parent
control and ablation for every enabled branch.
