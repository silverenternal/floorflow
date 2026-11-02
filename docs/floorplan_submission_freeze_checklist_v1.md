# Floorplan Submission Freeze Checklist v1

Status: ready for use after the paused predicted-model search is resolved.

## Preconditions

- Select one explicit checkpoint; do not freeze a directory of candidates.
- Commit the selected model code, operation parser, evaluators, and thresholds.
- Require `git status --porcelain` to be empty.
- Obtain a case manifest that has not been inferred or evaluated by any current
  predicted model.
- Verify that no test-derived metric has been used for checkpoint selection.

## Freeze Inputs

- selected checkpoint;
- predicted-control generation config;
- held-out prompt manifest generated from case IDs only;
- taxonomy;
- operation-success contract;
- FEVP, graph, and structural evaluator source hashes;
- metric tolerances and thresholds.

## Required Command Shape

```bash
uv run python scripts/prepare_predicted_control_submission_freeze_v1.py \
  --checkpoint <selected.pt> \
  --config <selected-config.json> \
  --prompt-manifest <heldout-case-ids.json> \
  --taxonomy outputs/cache/cubicasa5k_full_v2/taxonomy.json \
  --operation-contract configs/predicted_operation_success_contract_v1.json \
  --heldout-manifest <heldout-manifest.json> \
  --output configs/predicted_control_submission_freeze_v1.json
```

The command refuses a dirty worktree.

## One-Shot Rules

1. Run the frozen model on the new split once.
2. Do not change thresholds, checkpoint, prompt filtering, or evaluator after
   seeing outcomes.
3. Retain semantic failures, graph failures, abstentions, and negative metrics.
4. Recompute operation success, locality, validity, plausibility, fidelity, and
   graph diagnostics from the same rows.
5. Replace candidate paper columns with the frozen result.
6. Re-run integrity, figure, PDF, anonymity, and artifact checks.

## Current Blocker

No strictly fresh CubiCasa test subset is available: all 503 test cases were
inferred by the predicted completion run. A new compatible held-out manifest is
required before final submission.
