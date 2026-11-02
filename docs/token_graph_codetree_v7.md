# FloorFlow Token-to-Graph and CodeTree v7

## Scope

This revision implements the next structural generation layer. It is
CPU-validated and makes no GPU or SOTA claim.

## Token-to-Discrete-Graph Residual

`TokenGraphResidualHead` uses entity-specific room/wall/opening/junction
queries and room-pair edge reasoning to apply bounded residuals to:

- room existence and room class;
- wall existence and wall type;
- opening existence and opening host;
- room adjacency;
- junction degree.

Junction coordinates and degree have direct targets. Inference projects wall
endpoints onto active token junctions before canonical assembly. Graph residuals are
confidence-gated: if token confidence is below the refiner safety threshold,
the discrete graph gate is zero and vector-only topology is retained.

## Two-Level Room/Polygon Code

`CodeTreeAnchorDecoder` implements the second level of a TLC-Plan/Raster2Seq
style hierarchy:

1. room-stage token context forms room-level queries;
2. learnable polygon anchors provide the local shape code;
3. a decoder predicts bounded anchor residuals.

CodeTree polygons are Hungarian-matched against canonical room polygons and are
added as an actual candidate geometry in consumer reranking.

## Beam Search and Verifier Reranking

`token_beam_search.py` implements grammar-constrained beam search using the
cached self/cross attention KV state.

Each completed beam is passed through token-to-vector refinement and the
canonical/watertight contract. Candidate scores combine:

- contract validity;
- functional watertightness;
- room-count error;
- hole count;
- model sequence log-probability;
- Space Syntax public-integration advantage.

If a generated/codetree candidate is invalid or less watertight than vector-only,
the consumer applies per-sample rollback to vector output.

## Multi-step Rollout and GRPO

`token_rollout_steps` supports multiple self-prediction passes before the
refiner consumes token state.

`token_grpo.py` provides group-relative advantage normalization, sequence
log-probability policy loss, and optional reference-logit KL. It is a
CPU-tested post-training interface; it is not promoted without controlled
rollout evidence.

## Immutable Design Points

`design_points.py` derives room/wall/opening design points from an edit region,
marks untouched entities immutable, and produces an immutable token mask.
Grammar-constrained generation can force immutable token positions from a
reference sequence, including beam search and the two-pass inference API.

## Space Syntax

`space_syntax.py` computes graph integration, public integration advantage, and
connectivity. It is used as a candidate reward term together with a
deterministic validity/watertight/room-count reward.

## Public Asset Audit

- Raster2Seq: official code is available under `third_party/Raster2Seq`; the
  anchor-residual design was reused without importing its CUDA/detectron stack.
- CE2EPlan: code/data are available under `.data/external/CE2EPlan`; boundary
  and room alignment losses were reviewed for compatibility with existing
  vector losses.
- TLC-Plan: only README/assets are present, so only the two-level CodeTree
  concept was used.
- HouseMind, FMLM, GraphCAD 2026, and RenoLayout do not expose complete official
  implementations in the audited sources, so no reproduction is claimed.

## CPU Contract

Report:
`outputs/train/floorflow_token_graph_codetree_contract_cpu_v9/report.json`.

Observed:

- graph residual head and CodeTree head are present in the checkpoint;
- entity-specific graph residual, junction-coordinate loss, and Hungarian
  CodeTree loss are finite and reported;
- random model confidence triggers refiner and graph fallback;
- CodeTree remains an auxiliary branch with finite anchor residual;
- two-step rollout and room-instance semantic tokens were enabled;
- conflict projection still triggers for wall/opening;
- consumer contract is fail-closed when all branches are invalid.

## Verification

- Non-chip suite including the repository-wide line guard: `890 passed`.
- Changed files pass Ruff.
- All non-chip Python files remain below 600 lines.
- `uv build` succeeds.

## Claim Boundary

This establishes the executable discrete graph consumer, two-level code branch,
verified candidate reranking, policy-optimization interface, immutable design
points, and Space Syntax signal. It does not establish improved validity,
watertightness, Room IoU, GED, or external SOTA.
