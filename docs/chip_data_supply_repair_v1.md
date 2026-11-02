# Chip Data Supply and Validation Feedback Repair v1

Date: 2026-10-31

Status: CPU implementation and audit complete. No GPU experiment was run.

## Implemented

- Frozen validation curve independent of the training seed.
- Complete train/dev/range/config/schema fingerprints.
- Legalized checkpoint shortlist ranking and selection report.
- Legalizer/refiner displacement-rung sidecar format and training attachment.
- Vectorized padded edge gather in `PackedCaseStore`.
- CPU batch materialization prefetch within the gradient-accumulation window.
- Parallel CPU token OT matching for training validation.
- Multi-dimensional complexity sampler across block count, b2b degree and pin
  density.
- Mixed random/shelf source mode with fixed random and shelf controls.
- Optional geometry augmentation probability with `none` control.
- Audited 21-120 range cache with matched 20k exposure to the legacy ladder.

## Data Evidence

- train: 50,000 cases, 21-60 blocks.
- dev: 12,000 cases, 21-120 blocks.
- dev cases above the training maximum: 7,212, or 60.1%.
- shared eval curve: 500 cases, 5 per block count, eval seed 0.
- range cache: 200,000 cases across 21-120 blocks.
- range matched subset: 20,000 cases, 12,001 above 60 blocks.

## Artifacts

- `configs/chip_data_manifest_50k_v1.json`
- `configs/chip_eval_curve_v1.json`
- `outputs/reports/chip_data_supply_audit_v1.json`
- `scripts/audit_chip_data_supply.py`
- `scripts/build_chip_response_sidecar.py`
- `scripts/select_chip_checkpoint_shortlist.py`

## Boundaries

The implementation proves cache integrity, sampler behavior, sidecar contracts,
assignment equivalence and checkpoint ranking rules. It does not establish a
quality improvement or run the range training arm.
