# Room Page-Floor Five-Seed Confirmation Results v1

Date: 2026-11-01

## Decision

The promoted room mechanism passes the full-train five-seed gate:

```text
page-floor crop
+ canonical parent warm-start
+ encoder adaptation
+ query-set head
+ connected query-count selection gate
```

Freeze this architecture and checkpoint-selected readout for downstream
wall/junction integration. This is a relative paired result, not a claim that
absolute Room IoU has reached the final CAD target.

## Protocol

- Train: `5,122` valid CubiCasa floors.
- Calibration: val offset `0`, `384` cases.
- Disjoint holdout: val offset `384`, `107` cases.
- Full validation: all `491` cases after threshold/readout freeze.
- Seeds: `19, 23, 29, 31, 37`.
- One canonical warm-start for every seed.

## Holdout Paired Effects

| Metric | Mean delta | Seed range | Improved seeds | CI-excluding seeds |
|---|---:|---:|---:|---:|
| Room IoU | +0.095561 | [+0.086502, +0.110607] | 5/5 | 5/5 |
| Instance F1 | +0.143787 | [+0.128665, +0.153932] | 5/5 | 5/5 |
| Count MAE | -3.203738 | [-3.457944, -3.018692] | 5/5 | 5/5 |
| Type accuracy | +0.129393 | [+0.119626, +0.138941] | 5/5 | 5/5 |

Case-clustered pooled holdout evidence:

| Metric | Delta | 95% CI | Wilcoxon p | Rank-biserial |
|---|---:|---:|---:|---:|
| Room IoU | +0.095561 | [0.073985, 0.116945] | 2.63e-12 | +0.7865 |
| Instance F1 | +0.143787 | [0.114442, 0.172838] | 1.24e-13 | +0.8413 |
| Count MAE | -3.203738 | [-3.975701, -2.416822] | 3.54e-11 | -0.7515 |
| Type accuracy | +0.129393 | [0.049392, 0.209239] | 2.24e-3 | +0.4263 |

The pooled test clusters repeated seed rows by case. It is descriptive and does
not turn five seeds into independent population samples.

## Full Validation

At each seed's frozen selected blend, mean full-val491 values are:

| Metric | Candidate mean | SD | Range | Parent mean | Paired delta |
|---|---:|---:|---:|---:|---:|
| Room IoU | 0.310138 | 0.008438 | [0.302635, 0.320101] | 0.220191 | +0.089948 |
| Instance F1 | 0.247189 | 0.006372 | [0.239678, 0.254476] | 0.135731 | +0.111458 |
| Count MAE | 4.423218 | 0.351282 | [4.030550, 4.971487] | 7.811 | -3.387780 |
| Type accuracy | 0.640486 | 0.011324 | [0.632380, 0.660072] | 0.489783 | +0.150703 |

Room support IoU is unchanged at `0.753083`.

## Boundary

The improvement is robust and paired-significant, but absolute Room IoU
`0.31` remains below the next `0.45` internal quality target. The accepted
mechanism should now feed wall, junction, and adjacency consumers; new room
quality work must target the remaining absolute gap rather than another
unconsumed residual head.
