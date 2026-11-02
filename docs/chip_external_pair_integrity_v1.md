# Chip External Pair Integrity v1

Status date: 2026-10-30.

## Purpose

External candidate/control PPA comparisons are valid only when the two arms
produce different legalized macro placements. If the legalizer collapses both
proposals to the same TCL, downstream differences measure solver noise rather
than the model.

The machine audit is:

```text
scripts/audit_chip_external_pair_integrity.py
```

It compares SHA-256 hashes of the actual legalized TCL files used by ORFS.

## Results

### v44

`outputs/reports/chip_v44_external_pair_integrity.json`

- 8 designs.
- 18 design-seed pairs.
- 0 collapsed pairs.
- The v44 negative routed-PPA conclusion remains a valid model comparison.

### v45

`outputs/reports/chip_v45_external_pair_integrity.json`

- 15 cases.
- 13 completed cases.
- 7 collapsed candidate/control cases.
- 13 completed cases contain only 8 valid distinct pairs.

Collapsed cases:

- `asap7/coralnpu`
- `asap7/swerv_wrapper`
- `nangate45/ariane133`
- `nangate45/ariane136`
- `nangate45/bp_fe_top`
- `nangate45/swerv_wrapper`
- `sky130hd/microwatt`

Five of these collapsed pairs completed downstream flow:

- `nangate45/ariane133`
- `nangate45/ariane136`
- `nangate45/bp_fe_top`
- `nangate45/swerv_wrapper`
- `asap7/swerv_wrapper`

They cannot support a candidate/control model comparison even though ORFS
finished. The original endpoint reports remain archived.

Restricting to the eight completed distinct pairs changes the external conclusion
from the polluted all-completed mean of `+0.873%` routed wirelength to
`+1.408%` mean (`+1.760%` median), with 3 wins and 5 losses. The corresponding
machine report is:

`outputs/reports/chip_v45_external_valid_pairs_summary.json`

This remains a negative v45 model result and cannot be turned positive by the
corrected sensitivity cohort.

## Corrected Sensitivity Cohort

`outputs/reports/chip_v45_collapse_sensitivity.json`

The legalizer now preserves a feasible zero-overlap `scale_pack` result when the
whole packing rectangle only needs a rigid translation into the core. The
corrected sensitivity run reconstitutes all seven pairs:

- 3 use `scale_pack_core_fit`.
- 1 uses `shelf_pack_fallback_positioned`.
- 3 use a common `shelf_pack_forced_positioned` route after the automatic routes
  differ within the case.

All 7 corrected candidate/control pairs have distinct TCL hashes and a common
route within each case. This cohort is diagnostic only.

Routed sensitivity is scheduled sequentially. `sky130hd/microwatt` runs first
after A17 and the M6 response queue finish; the remaining six pairs wait for
that job rather than starting concurrent OpenROAD flows.

The repaired layout cohort is assembled in:

`outputs/reports/chip_v45_repaired_cohort_manifest.json`

It contains eight original valid pairs plus seven corrected sensitivity pairs,
for exactly 15 unique cases. All 15 have distinct candidate/control TCL hashes
and a common route within each case. Eight routed endpoints are complete; seven
remain pending. This repaired observed cohort is not a fresh untouched holdout.

## Root Cause Boundary

The collapse occurs after generation and refinement: candidate/control model
outputs and pre-legalization structural metrics differ, but the selected
legalization route writes the same TCL. Both `shelf_pack_forced` and
`shelf_pack_fallback` cases are affected. The next diagnostic is the existing
position-aware shelf path on the same frozen pairs.

This diagnosis does not prove that all v45 routed differences are wrong. It
invalidates only the collapsed pairs and requires every future pair to pass the
TCL-hash gate before ORFS.

## Policy

1. Preserve the original v45 result and failures unchanged.
2. Label corrected reruns as a sensitivity cohort, not replacement evidence.
3. Require distinct TCL hashes before any candidate/control PPA run.
4. Use rigid core translation before shelf fallback when the scale-pack bounding
   box fits the core.
5. Do not treat identical downstream metrics as a model tie; they indicate
   legalized proposal collapse.
