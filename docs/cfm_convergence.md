# Fast CFM convergence contract

The joint contour-to-CAD trainer uses the maintained `torchcfm` implementation
instead of a local probability-path reimplementation. The default `fast_cfm`
preset matches our task: the source is the contour-conditioned pixel/element
distribution, the target is the binary foreground CAD state, and the probability
path is the standard conditional straight path with logit-normal time samples.
Exact full-resolution minibatch OT is deliberately disabled because its cost
matrix grows quadratically with the number of CAD pixels.

## Training controls

- `torchcfm.ConditionalFlowMatcher`, logit-normal time, `sigma=0.05`
- fused AdamW on CUDA with separate backbone and distribution-head learning rates
- bf16 autocast, persistent workers, prefetch and channels-last support
- gradient accumulation counted in optimizer updates, not micro-batches
- warmup-cosine applied before the first update and after every resume
- EMA evaluation/selection and finite-loss fail-fast
- optimizer, scheduler, EMA, epoch, global update and best validation metric are
  saved in `last.pt`

Every run writes `run_manifest.json` with the TorchCFM version, probability path,
time distribution, effective batch size and update budget.

## Next training command

Only run this after the element-mask localization gate in `todo.json` passes.

```bash
uv sync --extra train --extra dev --extra vector
uv run python scripts/train_joint_distribution_cfm.py \
  --checkpoint outputs/train/condition_structured_full_v1_low_lr/best_room_iou.pt \
  --output-dir outputs/train/distribution_joint_full_fastot_v1 \
  --data-root .data/cubicasa5k \
  --cache-root outputs/cache/cubicasa5k_full_v2 \
  --flow-preset fast_cfm \
  --gradient-accumulation-steps 2 \
  --selection ema \
  --precision bf16
```

Resume with the same budget arguments and point `--resume` at `last.pt`. Do not
resume from `best.pt`, because `best.pt` contains EMA weights while the saved
optimizer state belongs to the raw training model.
