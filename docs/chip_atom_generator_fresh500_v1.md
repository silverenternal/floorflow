# Chip Atom-Generator Fresh500 Screen v1

Date: 2026-11-02

## Protocol

The fresh confirmation cohort contains 500 previously unused dev cases and is
block-count stratified with 100 cases in each bin: 21-40, 41-60, 61-80, 81-100,
and 101-120. It has zero overlap with the prior consensus and holdout cases.

Four frozen candidates plus the M48 seed21 candidate were inferred at NFE4,
refiner 400, seed 19, EMA, and the same projector/legalizer budget. The deployed
four-view selector was not modified. A separately trained five-view selector was
evaluated only as an atom-candidate diagnostic.

## Candidate-Pool Oracle vs the Four-View Reference

| Pool | JSR oracle | Strict structural oracle | Constraint-safe oracle |
|---|---:|---:|---:|
| Four candidates | 328/500 | 189/500 | 106/500 |
| Plus M48 seed21 | 362/500 | 220/500 | 123/500 |

The additional checkpoint adds 34 joint, 31 strict-structural, and 17
constraint-safe oracle cases. The CSJ target is 175 cases, so the pool still
misses at least 52 safe cases even with a perfect selector.

## Selected Five-View Screen

The existing five-view selector selected:

- JSR: 368/500;
- strict structural: 154/500;
- constraint-safe: 84/500.

The four-view selected diagnostic was 364, 151, and 84 respectively. The extra
checkpoint therefore adds only four JSR and three strict wins and no safe wins.
This is not a promotion. It confirms that candidate-pool generation must target
constraint-safe joint cases directly, not just add another checkpoint.

## Artifacts

- `configs/chip_eval_curve_fresh500_post_s8_v1.json`
- `outputs/reports/chip_s205_fresh500_manifest_v1.json`
- `outputs/reports/chip_s205_fresh500_5view_manifest_v1.json`
- `outputs/cohorts/chip_s205_fresh500_5view_v1`
- `outputs/reports/chip_s6_cluster_reflow_actions_fresh500_5view_v1.json`
