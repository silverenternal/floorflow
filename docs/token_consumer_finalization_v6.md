# FloorFlow Token Consumer Finalization v6

## Scope

This revision closes the final token-consumer audit items. It is CPU-validated
and makes no GPU or SOTA claim.

## Clean Validation

`forward_kwargs(..., training=False)` now disables token dropout, suffix
corruption, and rollout. Validation token CE, token accuracy, and geometry are
computed on clean teacher tokens, while training retains scheduled perturbation.

## Token-Implied Graph

- token logits are masked by valid token positions before graph aggregation;
- adjacency is combined with probabilistic OR, not clamped summation;
- room label, wall type, and opening host probabilities normalize only valid
  mass;
- graph tensors are registered as non-persistent model buffers and stay on the
  model device.

## Paired Consumer Evidence

The candidate configuration evaluates 8 validation cases by default. CPU smoke
uses a one-case CLI override for speed.

The report includes:

- vector-only, teacher-refined, and generated-refined contracts;
- paired validity and watertightness deltas;
- deterministic bootstrap 95% intervals;
- generation completion and 64-token efficiency completion;
- canonical validity and room-count errors;
- watertightness, hole count, dangling endpoint rate, boundary collisions, and
  outside-footprint pixels per branch.

Validity or watertightness regression can hard-stop training.

## Cached Decoding

`FloorplanTokenHead` now has an incremental KV cache:

- cached self-attention keys/values per decoder layer;
- cached cross-attention memory per decoder layer;
- incremental grammar state with cached vocabulary-kind ids;
- batched generated-token refinement from cached base outputs.

A unit test requires incremental logits to match full causal forward.

## Refiner Safety

- confidence temperature is learnable;
- confidence is aggregated per CAD stage;
- room, wall, and opening entities have separate gates;
- low confidence explicitly falls back to vector-only output;
- gates, entity gates, residual norms, confidence, and fallback are reported.

## Data Efficiency

- token lengths, canonical plans, and token targets are cached during bucket
  construction;
- the first dataset access reuses them rather than running canonicalization a
  second time.

## CPU Contract

Report:
`outputs/train/floorflow_token_consumer_contract_cpu_v6/report.json`.

Observed:

- a 163-token target used a 203-token dynamic generation budget;
- 203-token KV-cache generation completed inside the smoke run;
- token-implied adjacency, room label, wall type, and opening host terms were
  emitted;
- all three consumer branches reported validity and watertight diagnostics;
- random-initialized confidence was below the safety threshold, so refiner
  fallback was explicitly 1.0 and residual norms were zero.

These random-initialized results are diagnostics, not performance claims.

## Verification

- Non-chip suite including the repository-wide line guard: `885 passed`.
- Changed files pass Ruff.
- All non-chip Python files remain below 600 lines.
- `uv build` succeeds.

## Claim Boundary

This completes the executable token consumer, paired evidence, safety, and
inference-cost contract. It does not establish completion, validity,
watertightness, Room IoU, GED, or external SOTA improvements.
