# Bookshelf External Structural Holdout v1

Status date: 2026-10-30.

## Scope

The repository now contains a locally downloaded and hash-frozen external
structural cohort:

- GSRC HARD and SOFT variants: six block counts each.
- MCNC HARD and SOFT variants: five named designs each.
- 22 cases total.

This cohort evaluates behavior under the existing fixed-outline bookshelf
protocol. It is independent of FloorSet training data and is not routed-PPA
evidence. Results from this cohort cannot satisfy the ORFS external-PPA gate.

## Provenance

The files were downloaded through the pinned Trinity loader from the University
of Michigan mirrors documented in:

- `third_party/Trinity/src/trinity/floorplan/data/paths.py`

The site does not expose a clear redistribution license in this repository.
Therefore the local files are marked
`unverified_local_research_use_only`; do not copy them into a public artifact
package.

The machine manifest is:

`outputs/reports/chip_v46_bookshelf_holdout_manifest.json`

It records every source file hash, source URL, block count, b2b/p2b edge count,
terminal count, fixed-outline dimensions and reference-placement availability.

## Evaluation protocol

1. Use the pinned Trinity `with_fixed_outline` setup with `gamma=0.10` and
   `aspect=1.0`.
2. Evaluate candidate and control with the same NFE, refiner steps, legalizer,
   seed and scoring implementation.
3. Report HARD and SOFT separately because they have different shape freedom.
4. Report all 22 cases; do not select cases after seeing outcomes.
5. Keep these results separate from the ORFS routed-PPA denominator.

## Initial screen

`outputs/reports/chip_v46_bookshelf_external_smoke_d4.json` evaluates
`gsrc/hard/n10` and `mcnc/soft/apte` with four draws at NFE4/refiner400.

Both candidate and control produced zero outline-feasible draws. This is an
adaptation failure under the fixed-outline protocol, not a model ranking.
Do not report the raw HPWL ratio as a win or loss from this smoke. A larger draw
count or an explicit outline-adaptation mechanism is required before using this
cohort as positive evidence.

The 16-draw follow-up
`outputs/reports/chip_v46_bookshelf_external_smoke_d16.json` also produced zero
feasible draws for both arms. The failure is therefore not explained by the
small draw budget. Do not expand this evaluator to the full 22-case cohort until
fixed-outline legalization or outline conditioning is repaired.

Increasing the existing outline refiner weight to 10 at 16 draws and to 100 at
four draws also produced zero feasible layouts. This rejects a simple
outline-weight sweep and identifies an absent L0 fixed-outline conditioning
path: the model predicts a bounding box but does not receive the desired
bookshelf outline as a conditioning variable.

The smallest proposed architecture-aligned fix is specified in
`docs/chip_fixed_outline_conditioning_v1.md`.
