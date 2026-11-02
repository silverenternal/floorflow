# Chip Full-Data Chain Review v1

Date: 2026-10-30

Scope: FloorSet packed cache -> full-data training/resume -> EMA checkpoint
selection -> checkpoint loading -> Official-100/refiner/legalizer evaluation ->
derived statistics and reports.

No full training arm was started during this review.

## Fixed defects

### 1. Validation changed the training mode

`train_evaluation.evaluate()` and the legacy `train_chip_large.evaluate()` called
`model.eval()` without restoring the caller's mode. Because the default eval
schedule includes step 0, exploratory training could continue in eval mode after
the first validation.

Fix: both evaluators restore the previous `model.training` value.

### 2. Best checkpoint selection used a different state than deployment

Training curves evaluated the raw model, while official evaluation loads
`--use-ema`. A step selected as best for raw weights could therefore be
suboptimal for the EMA checkpoint actually evaluated.

Fix: `train_chip_slot.py` now supports `--eval-state {ema,raw}` and defaults to
`ema`. The selected state and checkpoint interpretation are recorded in the
training report.

Historical training reports that predate this field contain raw-model curves and
must not be reinterpreted as EMA-selection evidence.

### 3. Resume did not restore the full stochastic state

The checkpoint saved CPU torch/Python/NumPy RNGs but omitted CUDA RNG and the
balanced-stream sampler queue/cursor/rng.

Fix:

- save and restore the active CUDA generator;
- serialize `BalancedBatchSampler`;
- fail fast if a balanced-stream resume checkpoint lacks sampler state;
- reject resume attempts that change architecture, sampling mode or eval state.

Older checkpoints remain loadable for ordinary evaluation. Exact balanced-stream
resume from an older checkpoint is intentionally rejected.

### 4. Checkpoint/report replacement was not crash safe

Direct `torch.save` and `write_text` can leave a partial target if the process
dies during the write.

Fix: training checkpoints and curriculum reports now use same-directory temporary
files, fsync and atomic `os.replace`.

### 5. Exploratory runs could silently use full data

`train_chip_slot.py` previously treated an absent `--train-indices` as full-data
training.

Fix: exploratory training requires a frozen subset; full-data training requires
the explicit `--allow-full-data` override. The legacy full-data
`train_chip_large.py` requires the same override.

### 6. Train/dev overlap was not checked at use time

Split construction was intended to be disjoint, but consumers did not verify it.

Fix: training validates `(source path, source layout)` pairs before loading the
model. The audited 50k/12k cache has zero overlap.

### 7. Packed-cache corruption could fail late or silently

The mmap store did not validate required arrays, metadata counts, feature width,
padded block width or flattened edge offsets.

Fix: `PackedCaseStore` now rejects inconsistent metadata/count/offset/shape
layouts at construction.

### 8. Official case reports dropped legality flags

The Official-100 evaluator removed every key ending in `_feasible` when saving
case rows.

Fix: raw and legalized feasibility are retained in the per-case artifact.

### 9. Refiner batches replayed one noise sample

The refiner evaluator reset its generator to the same seed for every batch,
reusing the same noise vector sequence across different batches of the same block
count.

Fix: each batch uses a stable seed derived from the global inference seed and the
batch's first frozen case index. The same batch receives the same noise across
NFE settings, while different batches do not.

### 10. Paired statistics had wrong direction and weak alignment

`paired_delta()` counted positive deltas as improvements regardless of metric
direction and paired rows positionally.

Fix:

- added explicit `lower_is_better`;
- all chip report call sites mark cost/gap/V_rel/etc. as lower-is-better;
- unique case rows are aligned by `index`;
- missing/mismatched case indices and repeated-index order mismatches are
  rejected.

The numeric cost/V_rel means and bootstrap intervals were unchanged for the
audited reports, but incorrect `improved/worsened` counts were corrected.

### 11. Checkpoint loader ignored configured feature width

The legacy flat-model fallback always constructed `feature_dim=10`.

Fix: it now reads `config["feature_dim"]`.

## Validation

- Full pytest suite after the adaptive-selector cross-validation, matched-component decomposition, ORFS supply scan, outline MILP fallback, fallback on/off mechanism test, pooled outline seed statistics, outline failure-mode attribution, capacity FLOP preflight, capacity budget planning, queued gated capacity runner with refiner1600 diagnostics, S1-report validation, dynamic confirmation-horizon resizing, M6 alias repair/retry queue, current-evidence gating, active-task reference auditing and room-mask split: `962 passed`.
- Scoped Ruff checks: pass.
- Maximum first-party Python file length: `599` lines.
- Formal 2-step balanced-stream training smoke: pass.
- Resume from atomic checkpoint with sampler and RNG state: pass.
- Step-0 validation followed by training in train mode: pass.
- Official-100 tiny-model/NFE1 legalized smoke: pass.
- Refiner NFE1/refine0/refine2 smoke: pass.
- GPU architecture/protocol audit on Official-100: pass.
- v45 three-seed, matched-budget and 1600-step refiner reports regenerated from
  existing raw case artifacts; numeric endpoint values remain unchanged.

## Remaining boundaries

- Existing pre-review checkpoints do not contain CUDA or balanced sampler state.
- Existing pre-review training reports remain raw-evaluation evidence if they
  lack `eval_state`.
- The smoke evaluations use a tiny diagnostic checkpoint and are plumbing tests,
  not quality evidence.
- A future formal full-data finalist must pass `--allow-full-data`, use
  `--eval-state ema`, preserve the frozen subset provenance and run against its
  same-seed matched control.

