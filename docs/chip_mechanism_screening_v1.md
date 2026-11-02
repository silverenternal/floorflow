# Chip Mechanism Screening v1

## Purpose

Mechanism iteration must not spend full-data or full-holdout compute before there is
evidence that the mechanism changes the intended behavior. The screening tier is
diagnostic and is never reported as final holdout or SOTA evidence.

## Frozen Subsets

- Training: `configs/chip_train_screen_5k_v1.json`.
- Training cases: 5,000, deterministically stratified across all cached block counts.
- Training seed: 23.
- Quick-screen budget: 2,000 optimizer steps with refiner400.
- Confirmation budget: 5,000 optimizer steps with refiner1600.
- Legalized validation: the frozen 32-case manifest in
  `outputs/reports/chip_dev_screen_v45.json`.
- Routed-PPA dependent mechanisms: the frozen three-case external screen in
  `configs/chip_v45_postprocess_v2_screen.json`.
- Mechanism config: `configs/chip_mechanism_screen_v1.json`.

The training-subset manifest records its cache metadata hash, selection seed, block-count
histogram and source-family coverage. It must not be regenerated after seeing outcomes
for the same screen.

## Promotion Flow

1. Train one fixed seed with `--train-indices configs/chip_train_screen_5k_v1.json`.
2. Evaluate legalized metrics on the frozen 32-case dev manifest.
3. If the claim concerns routed-PPA transfer, run the frozen three-case screen.
4. Keep every failed screen in `chipstodo.json`; do not retune against full outcomes.
5. A quick pass is not a candidate. Confirm it with 5k steps and refiner1600 under the
   same subset and seed.
6. Register only a confirmation winner in `configs/chip_candidate_registry.json`.
7. Only a registered candidate may receive 20k training or the complete 15-case holdout.
8. The first full-data run must use the same training seed as the registered subset
   candidate and a matched full-data control. A subset win is never copied forward as
   a full-data result.
9. If the matched full-data run fails, mark the candidate rejected in the registry and
   retain both the positive subset result and the negative full-data result.

Two graph variants (`typed_layers=2` and per-type attention bias) improved the
2k/refiner400 screen but reversed at 5k/refiner1600. This observed optimism is why quick
passes cannot be reported as candidates or promoted directly.

## Candidate Registry

The registry separates active candidates from rejected screens. A candidate record must
contain the exact checkpoint, training subset, seed, inference protocol, legalized
metrics, optional routed metrics, and report paths. A registry entry is not a SOTA claim.

## Artifact Retention

Keep the JSON reports and every checkpoint referenced by a report, registry, manifest, or
TODO. Completed arms may discard their rolling `last.pt`; retain `best.pt` or `final.pt`
according to the recorded inference protocol. Never delete the active arm's rolling
checkpoint. Raw routed-PPA runs are separate from training artifacts and require their
own retention audit before cleanup.

## Reporting Boundary

Subset screens answer whether a mechanism is worth more compute. They do not establish
generalization, seed stability, routed-PPA improvement, or publication readiness. Those
claims still require the frozen full gate.

The 32-case legalized dev manifest and the three-case routed screen are frozen mechanism
diagnostics. Neither set may be used to select a checkpoint, subgroup, or inference
hyperparameter for the final claim.
