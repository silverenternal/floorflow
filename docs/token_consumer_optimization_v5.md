# FloorFlow Token Consumer Optimization v5

## Scope

This revision closes the remaining teacher-forcing, generated-token, graph, and
batching gaps in the FloorFlow token consumer. It is CPU-validated and makes no
GPU or SOTA claim.

The final clean-validation, watertightness, graph-buffer, and KV-cache
follow-up is documented in `docs/token_consumer_finalization_v6.md`.

## Generation Budget

Formal constrained generation now uses a target-length-derived budget:

```text
min(max_length, generation_max_new_tokens,
    ceil(target_token_length * generation_budget_multiplier))
```

The default multiplier is 1.25. A separate 64-token efficiency completion rate
is retained, so correctness and budget efficiency are no longer conflated.

## Paired Consumer Validation

For the same validation cases, the contract now evaluates:

1. vector-only output;
2. cached-base output refined with teacher token ids;
3. cached-base output refined with generated token ids.

All branches use the same canonical decoder and validation gates. The report
includes contract validity, token round-trip, room-count error, generation
completion, efficiency completion, and generated-minus-vector validity delta.
Checkpoint selection rewards generated contract validity and penalizes validity
regressions.

## Exposure-Shift Mitigation

Training token input differs from the loss target:

- token dropout;
- contiguous suffix corruption;
- scheduled corruption ramp;
- one-step self-prediction rollout before the final token state is consumed.

The token CE remains supervised against ground-truth targets.

## Gated Refinement

`TokenVectorRefiner` now exposes:

- token confidence;
- separate room, wall, and opening gates;
- room/wall/opening residual norms;
- bounded channel-specific residual updates.

Teacher and generated refinements are hard-gated when they regress canonical
validity relative to vector-only output and generated completion is available.

## Token-Implied Soft Graph

Token logits are converted into soft graph estimates:

- room adjacency from `ADJACENCY` token probabilities;
- room labels from `ROOM_BEGIN` token probabilities;
- wall type from `WALL` token probabilities;
- opening host from `OPENING` token probabilities.

These distributions are compared directly with vector adjacency, room-class,
wall-type, and opening-host probabilities. This is distinct from the earlier
target-to-vector typed graph loss.

## Length Index and Inference Cache

- Token lengths are cached during length-bucket construction.
- The corresponding plan and token target are reused by the first item access.
- Two-pass inference computes the base output once.
- The second pass runs only token head and refiner from cached base outputs.
- Generated-token inference supports per-sample completion inside a batch.

## CPU Contract Evidence

Report:
`outputs/train/floorflow_token_consumer_contract_cpu_v5/report.json`.

Observed:

- dynamic generation budget produced a 203-token attempt from a 163-token target;
- efficiency-budget completion remained separately reported as zero;
- both token-structure conflict pairs were projected;
- all four token-implied soft graph terms were emitted;
- vector/teacher/generated contracts were evaluated as paired branches;
- scheduled sampling, prefix corruption, and one-step rollout were active;
- no validity regression was hidden when all three branches were invalid.

Random-initialized completion and validity remain near zero and are reported as
diagnostics, not model claims.

## Verification

- Non-chip suite including the repository-wide line guard: `883 passed`.
- Changed files pass Ruff.
- All non-chip Python files remain below 600 lines.
- `uv build` succeeds.

## Claim Boundary

This establishes the executable generated-token consumer and paired evaluation
contract. It does not establish improved completion, validity, watertightness,
Room IoU, GED, or any SOTA result. Those require controlled GPU training and
held-out evaluation with this contract frozen.
