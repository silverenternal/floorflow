# Chip Route-Transfer Provenance v2

Date: 2026-11-01

## Decision

The external routed-PPA confirmation track is blocked. Existing outcomes cannot
be promoted to a fresh 15-20 design holdout:

- `v44` routed calibration covers 8 independent designs and 18 design-seed
  measurements;
- only 10 of 18 calibration measurements preserve the complete ORFS command
  recipe in the combined report;
- the later `v45` cohort and the ChiPBench adapter cohort have already been
  opened, so they are observed/sensitivity evidence, not fresh confirmation;
- the refreshed ChiPBench pair manifest finds 13 of 15 candidate/control TCL
  pairs collapsed to byte-identical legalized placements;
- only `ethernet` and `isa_npu` retain distinct ChiPBench legalized pairs, and
  neither has a complete symmetric routed-PPA pair;
- after source-family deduplication, only 2 static ORFS macro designs remain
  outcome-unopened: `gf180/aes-hybrid` and
  `ihp-sg13g2/i2c-gpio-expander`.

The current gap to a 15-design confirmation cohort is therefore 13
source-disjoint designs. Repeated seeds count as measurements, not designs.

## Frozen Provenance Contract

Every candidate/control pair must record:

- canonical design/source family;
- platform and config SHA-256;
- proposal/checkpoint identity;
- model and legalizer seed;
- candidate/control legalized TCL SHA-256;
- distinct-TCL, feasibility, overlap and common-route checks;
- ORFS commit and pinned image manifest/config digests;
- exact target, `OR_SEED`, `GRT_SEED`, detailed-route arguments and route
  recipe SHA-256;
- complete routed, congestion, DRC, timing, power and area endpoint metrics.

The ORFS staging manifest now fails closed when any placement pair is collapsed,
illegal, overlapping or route-mismatched. Previously it could stage the old
15-design inventory after those checks had failed.

## Evidence

Machine-readable evidence:

- `outputs/reports/chip_route_transfer_provenance_v2.json`
- `outputs/reports/chipbench_holdout_pair_manifest_v1.json`
- `outputs/reports/chipbench_holdout_orfs_manifest_v1.json`
- `outputs/reports/chip_v44_external_pair_integrity.json`
- `outputs/reports/chip_v45_external_pair_integrity.json`

The provenance report is deliberately `blocked`, not `pass`: a complete
machine-readable manifest is not a substitute for an unopened routed cohort.

## Minimal Path to Confirmation

1. Repair the 13 collapsed ChiPBench legalized pairs and rerun the pair-integrity
   gate before any ORFS launch.
2. Select 15 source-disjoint, outcome-unopened macro designs. The two remaining
   static ORFS candidates are insufficient, so additional public macro-capable
   designs must be sourced or generated.
3. Freeze candidate/control placement hashes, config hashes, seeds and route
   recipes before seeing PPA.
4. Route one diagnostic pair first. Reject the cohort if candidate/control
   collapse, common-route mismatch, GRT failure or missing hold metrics occurs.
5. Only after the diagnostic passes, route the remaining 14-19 designs with the
   same pinned image and recipe.

Until those steps complete, the external result remains negative transfer and
cannot support routed-PPA SOTA or surrogate-to-PPA claims.
