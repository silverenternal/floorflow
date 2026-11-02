# Typed-Channel GPU Runtime Audit v1

Date: 2026-11-02

## Problems Found

1. The trainer defaulted to `model_predicted_full_v1`, while the current
   full491 predicted-control protocol uses `model_predicted_full_v3`.
2. Component logging called `.detach().cpu()` independently for loss, stage
   diagnostics, and roughly 30 objectives on every training batch.
3. Input transfers used blocking `.to(device)` calls despite pinned batches.
4. The first long screen crashed because `routing` and `room_adjacency` were
   logged but absent from the initial component-total dictionary.
5. A proposed GPU Sinkhorn replacement for exact query matching was benchmarked
   before promotion.

## Implemented Repairs

- The typed trainer now defaults to
  `outputs/control_contracts/model_predicted_full_v3`.
- Scalar diagnostics accumulate on `cuda:0` and transfer to the host once per
  epoch.
- Input tensors use `non_blocking=True` with pinned CUDA batches.
- Every logged objective has a stable accumulation key.
- Stage diagnostics use the same GPU accumulation path.

## Matching Decision

At shape `[4, 32, 32]`, the reproducible benchmark records exact CPU assignment
at `0.098542 ms` and GPU Sinkhorn at `1.634571 ms`, a `16.59x` GPU overhead. The
exact path is therefore retained: it is both faster and semantically exact at
the actual room-query size. GPU conversion is reserved for batch-wide scalar
aggregation, where the previous implementation introduced dozens of
synchronization points.

Reproduce with:

```bash
uv run python scripts/benchmark_assignment_paths_v1.py \
  --device cuda:0 \
  --output outputs/reports/typed_channel_assignment_microbench_v1.json
```

## Verification

- Short CUDA trainer smoke: `8 train / 8 val / 1 epoch` completed.
- Query, graph, room-mask, trainer-contract, and component-accumulation tests
  pass.
- No chip or NLP code was modified.

These are engineering improvements. They do not establish a metric gain or a
SOTA claim.
