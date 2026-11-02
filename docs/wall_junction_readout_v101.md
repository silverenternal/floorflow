# Wall / Junction Readout v101

Date: 2026-11-01

## Canonical Protocol

All numbers here use the explicit raster plan that produced the registered
Wall precision/recall baseline. Owner-cell-plan diagnostics are retained
separately but are not mixed with the canonical table.

## False-Positive Decomposition

With unconditional room seams disabled, calibration64 has:

| Metric | Value |
|---|---:|
| Wall precision | 0.8172 |
| Wall recall | 0.3953 |
| Wall F1 | 0.5071 |
| Wall continuity | 0.9646 |
| False-positive pixels | 1292.48 |
| True-positive pixels | 962.13 |
| False-negative pixels | 2116.66 |

False positives partition as:

| Provenance | Fraction |
|---|---:|
| GT-wall extension | 0.5790 |
| exterior duplication | 0.2419 |
| room-boundary fragment | 0.0904 |
| unsupported free wall | 0.0887 |

The dominant issue is not unsupported wall generation. It is wall-band
extension around valid wall centerlines.

## Wall-Gated Seam

`wall_gated` keeps a room seam only where the wall head has at least `0.2`
evidence. It is the first wall-readout candidate to improve both matched
splits over no-seam:

| Split | Wall F1 | Precision | Recall | Continuity |
|---|---:|---:|---:|---:|
| cal64 | 0.5252 | 0.7988 | 0.4177 | 0.9602 |
| holdout64 | 0.5303 | 0.8135 | 0.4217 | 0.9772 |

Paired Wall F1 delta:

- cal64: `+0.01810`, 95% CI `[0.00504, 0.03160]`;
- holdout64: `+0.01992`, 95% CI `[0.00794, 0.03258]`.

Recall improves with a CI above zero on both splits. Precision decreases by
about 0.02 but remains above `0.79`, well above the final `0.75` floor.
Continuity remains above `0.96`.

## Decision

- Promote `wall_gated` threshold `0.2` as the canonical evaluation candidate.
- Keep `no_seam` as the matched control.
- Reject `legacy` as the default because it unions every room seam into walls.
- Do not treat this as a full-test SOTA claim. Wall F1 remains below the staged
  `0.58` target.

## Remaining Work

Reduce GT-wall extension without losing recall. Candidate mechanisms:

- wall-band width calibration against the vectorized target;
- endpoint/junction-conditioned wall termination;
- owner-boundary provenance for interior candidate endpoints.

Evidence:

- `outputs/reports/wall_false_positive_decomposition_canonical_cal64_v102.json`
- `outputs/reports/wall_false_positive_decomposition_canonical_holdout64_v102.json`
- `outputs/reports/wall_junction_mechanism_wallgated_canonical_cal64_v102.json`
- `outputs/reports/wall_junction_mechanism_wallgated_canonical_holdout64_v102.json`
- `outputs/reports/wall_junction_mechanism_screen_v101.json`
- `outputs/reports/wall_gated_seam_promotion_v102.json`
