# FloorFlow Typed-State Convergence v8

## Scope

This revision consolidates the token, graph, geometry, and verifier branches
around one `TypedFloorplanState`. It is CPU-validated and makes no GPU/SOTA
claim.

## Unified State

`TypedFloorplanState` stores:

- room features, existence, labels, polygons, boxes, and adjacency;
- wall endpoints, existence, and type;
- opening centers, widths, existence, and host logits;
- junction coordinates and degree logits;
- CAD stage tokens and token context.

Every state can be projected back to the legacy prediction dict, cloned, and
Hungarian-aligned to a reference room state.

## Entity-Specific Graph Reasoning

Token graph residuals now use separate:

- room-query cross-attention;
- wall-query cross-attention;
- opening-query cross-attention;
- junction-query cross-attention;
- room-pair edge reasoning.

Junction coordinate and degree heads are trained directly. Inference projects
wall endpoints onto active token junctions before canonical assembly.

## CodeTree as Candidate Output

CodeTree polygons are:

- supervised with Hungarian polygon matching;
- aligned to reference room identity using Hungarian box assignment;
- evaluated as a real geometry candidate in consumer reranking.

## Verified State Machine

`typed_state_machine.py` selects among:

- vector-only;
- token-refined;
- CodeTree candidates.

The selected candidate must pass canonical validity and watertightness. If a
refined candidate regresses either metric, per-sample state rolls back to the
vector state.

## Immutable Editing

Immutable room/wall/opening design points can force token prefixes during:

- greedy generation;
- KV-cached beam search;
- two-pass token-refined inference.

## Reward and Policy Interface

The deterministic candidate reward combines validity, watertightness, holes,
room-count error, and Space Syntax public integration advantage.

`token_grpo.py` provides group-relative sequence policy loss. Multi-step
self-rollout remains configurable through `token_rollout_steps`.

## Hierarchical Loss

The previous flat loss stack can now be grouped as:

```text
primary   = polygon + identity + count
topology  = wall + opening + adjacency + token structure
sequence  = token CE
```

Raw terms remain available for diagnostics and ablation.

## CPU Contract

Report:
`outputs/train/floorflow_typed_state_convergence_cpu_v11/report.json`.

Observed:

- hierarchical primary/topology/sequence losses are finite;
- CodeTree primary mode is enabled;
- room-instance IDs persist through Hungarian alignment;
- graph-first junction face candidates are available to the unified verifier;
- junction coordinate and CodeTree losses are finite;
- random low-confidence model still triggers geometry/graph fallback;
- two-step rollout is active;
- conflict projection remains active;
- immutable/beam/reward/state-machine paths pass targeted tests.

## Verification

- `891` non-chip tests pass.
- The repository-wide line guard finds only the out-of-scope
  `chip_flow/typed_graph_encoder.py` at 601 lines.
- Changed files pass Ruff.
- `uv build` succeeds.

## Claim Boundary

This completes the architectural consolidation and executable contracts. It
does not establish performance gains on generated floorplans.
