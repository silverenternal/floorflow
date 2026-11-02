# FloorFlow Token Downstream Optimization v4

## Scope

This revision closes the token-branch downstream gaps identified after v3. It
is CPU-validated and does not produce a GPU or SOTA result.

## Training Chain

```text
canonical token targets
  -> grammar-masked token CE
  -> typed graph consistency
  -> token-weighted joint loss
  -> optional PCGrad-style conflict projection
  -> finite-gradient check and clipping
  -> AdamW parameter groups
```

## Token Consumer

`TokenVectorRefiner` consumes causal token hidden states through:

- room-query to token cross-attention for room polygon residuals;
- masked global token pooling for wall endpoint residuals;
- masked global token pooling for opening center and width residuals;
- bounded `tanh` updates with a default 0.02 residual scale.

Training uses teacher-forced token states. Inference supports two passes:

1. run the base model and expose `token_context_tokens`;
2. grammar-constrained decode a token sequence, then rerun the model with that
   sequence to obtain token-conditioned vector refinements.

`token_refined_generation` implements this two-pass inference wrapper.

## Token Readout

Validation now reports:

- overall token accuracy;
- per-kind F1 and macro F1;
- per-CAD-stage CE;
- one-step grammar-invalid rate;
- teacher-forced END accuracy;
- room/wall/opening token count errors;
- constrained generation completion and generated length;
- canonical contract validity when contract smoke is requested.

The checkpoint selection score includes token accuracy, macro F1, token CE,
grammar-invalid rate, structure consistency, and generation completion.

## Typed Graph Consistency

The consistency objective now covers:

- room, wall, and opening cardinality;
- room adjacency;
- opening host assignment;
- wall type;
- room labels;
- room vertex-validity counts;
- ring closure span;
- soft junction degree from predicted wall endpoints;
- bounded pairwise room-mask overlap.

## Optimization Controls

- token CE ramps in over a configurable warmup;
- typed structure loss begins after the CE warmup and ramps separately;
- `token_structure ↔ wall/opening` negative cosine terms are projected with a
  PCGrad-style conflict-aware backward;
- norm balancing updates more frequently;
- token modules use a separate learning-rate group;
- embeddings, norms, and biases use no weight decay;
- optimizer, token schedule, global step, vocabulary SHA, and balancer state are
  checkpointed and restored.

## Dynamic Sequence Batching

Token targets remain variable length until collate. Each batch is padded only
to its own maximum, and training uses `LengthBucketBatchSampler` to reduce
padding while retaining shuffling within length buckets.

## CPU Contract Evidence

Report:
`outputs/train/floorflow_token_downstream_contract_cpu_v4/report.json`.

Observed checks:

- both configured conflict pairs were detected with negative cosine and
  projected;
- normalized room partition excess is about 0.333 on the smoke batch;
- ring-closure consistency is produced;
- all token readout categories and generation diagnostics are emitted;
- dynamic collate and length bucketing execute on real RPLAN rows;
- contract and tokenization failures are fail-closed and reported;
- optimizer checkpoint contains parameter groups and AdamW state.

The random-initialized contract may be geometrically invalid. That is recorded
as evidence rather than hidden or used as a model claim.

## Verification

- Non-chip suite including the repository-wide line guard: `882 passed`.
- Changed files pass Ruff.
- All non-chip Python files remain below 600 lines.
- `uv build` succeeds.

## Claim Boundary

This establishes an executable, auditable downstream token-training and
inference interface. It does not establish improved Room IoU, GED, completion
rate, RPLAN SOTA, or full-CAD SOTA. Those require a controlled GPU training
screen and held-out evaluation with this contract frozen.
