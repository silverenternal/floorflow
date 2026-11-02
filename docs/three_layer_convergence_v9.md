# FloorFlow Three-Layer Convergence v9

## Scope

This is the final architecture-convergence pass. All previous heads remain
available; they are grouped through three typed state layers and one verified
transition interface. It is CPU-validated and makes no GPU/SOTA claim.

## Three Layers

1. `RoomInstanceLayer`
   - persistent room-instance IDs;
   - semantic room tokens;
   - existence and class.

2. `TopologyLayer`
   - room adjacency;
   - walls, openings, hosts, junction coordinates, and junction degrees.

3. `GeometryLayer`
   - room polygons;
   - wall endpoints;
   - opening centers and widths.

The existing identity decoder, vector decoder, graph head, token head, geometry
refiner, and CodeTree head are retained and write into these layers.

## Two-Level VQ CodeTree

`CodeTreeAnchorDecoder` now contains:

- a 32-entry room codebook;
- a 64-entry polygon codebook;
- straight-through vector quantization;
- room/polygon code indices;
- VQ commitment/codebook loss;
- the existing learnable anchor residual path.

CodeTree remains a primary geometry candidate with Hungarian room alignment.

## Identity and Graph Consistency

- room IDs persist through state reorderings;
- graph queries include room/wall/opening/junction ID embeddings;
- graph residuals remain entity-specific and confidence-gated;
- junction coordinates are projected into wall incidence before contract.

## Verified State Machine

One scored-state machine selects among:

- vector-only;
- token-refined geometry;
- CodeTree;
- graph-first junction faces.

Invalid or lower-watertightness candidates fall back to vector state.

## Loss and Policy Feedback

- hierarchical primary/topology/sequence loss remains active;
- VQ loss is attached to the CodeTree branch;
- GRPO and verifier-reward policy losses are available for replay batches;
- reward includes validity, watertightness, holes, room count, and Space Syntax.

## Immutable Editing

Immutable reference/mask constraints are passed through greedy generation,
beam search, and two-pass token-refined inference.

## CPU Contract

Report:
`outputs/train/floorflow_converged_contract_cpu_v14/report.json`.

Observed:

- hierarchy primary/topology/sequence losses are finite;
- CodeTree VQ loss is finite;
- verifier reward and policy loss execute inside the training batch;
- topology-first eval path is available;
- typed stage layers and room-instance ID state are emitted;
- the three-layer execution controller preserves all prior heads;
- room codebook inputs include room class and box;
- teacher branch reaches canonical validity and exact room count;
- generated greedy path remains fail-closed without completion;
- random low-confidence geometry/graph residuals fall back safely.

## Verification

- `892` non-chip tests pass, including the repository line guard.
- Changed files pass Ruff.
- `uv build` succeeds.

## Claim Boundary

The architecture is now converged and executable without removing head
functionality. No trained performance/SOTA claim is made.
