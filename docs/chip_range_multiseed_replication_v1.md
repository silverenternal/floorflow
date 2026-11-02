# High-Block Range Seed Replication v1

Date: 2026-10-30

Status: completed; rejected for seed stability.

## Protocol

The contrast compares a frozen 20k subset restricted to 21-60 blocks with a
frozen 20k subset spanning 21-120 blocks. All arms use 20k optimizer steps,
batch 128, projector 4, seed-matched initialization/data order, the same
legalizer and the frozen dev32 manifest.

Seed19 had completed previously. Seed20 was resumed from its valid step11999
checkpoint and completed to 20k. Seed21 was trained from scratch. All controls
and range arms reach legalized feasibility `1.0`.

## NFE4 / Refiner400

| Seed | Cost | V_rel |
|---:|---:|---:|
| 19 | -2.52% | -9.65% |
| 20 | -10.03% | -14.59% |
| 21 | -2.51% | +4.45% |

Pooled case-clustered result:

- cost `-5.10%`, 95% CI `[-14.54%, -0.76%]`;
- V_rel `-6.70%`, 95% CI `[-1.03%, +0.24%]`.

The cost improvement is stable by mean and pooled interval. V_rel is not.
Seed21 has severe 91-120 block regressions.

## NFE4 / Refiner1600

| Seed | Cost | V_rel |
|---:|---:|---:|
| 19 | -3.83% | -9.34% |
| 20 | -1.15% | +1.57% |
| 21 | -5.31% | -16.90% |

Pooled case-clustered result:

- cost `-3.46%`, 95% CI `[-7.57%, -1.06%]`;
- V_rel `-8.50%`, 95% CI `[-1.05%, +0.32%]`.

Seed21 passes the paired cost gate, but its V_rel interval still crosses zero.
Seed20 regresses V_rel. No seed passes both paired primary gates.

## Equal-Cost Ensemble Check

Pairwise equal latent averages at refiner400 did not help:

| Pair | Cost | V_rel |
|---|---:|---:|
| 19 + 20 | 1.24970 | 0.04808 |
| 19 + 21 | 1.27442 | 0.05162 |
| 20 + 21 | 1.26897 | 0.05521 |

The best pair improves V_rel `1.86%` over the seed20 single model but regresses
cost `3.31%` and doubles sampling NFE.

## Official-100 Transfer

All three range models were evaluated on the full official-100 endpoint at
NFE4/refiner1600 with the matched group-2 refiner:

| Model | Cost | V_rel |
|---|---:|---:|
| Trinity d192 matched anchor | 1.08803 | 0.02398 |
| Range seed19 | 1.17017 | 0.03042 |
| Range seed20 | 1.16768 | 0.03604 |
| Range seed21 | 1.18414 | 0.03785 |
| Range seed mean | 1.17399 | 0.03477 |

The seed mean remains worse than the matched anchor by `+7.90%` cost and
`+44.97%` V_rel. The dev32 cost-coverage gain does not transfer to the official
endpoint. This is a decisive rejection of the range data route as a SOTA fix.

## Decision

The 21-120 coverage strategy produces a reproducible cost reduction across all
three seeds, including at refiner1600. It does not produce seed-stable V_rel.
Do not register a range candidate. The next data/training test must target the
cross-seed V_rel tail rather than further broaden block-count coverage.

## Artifacts

- `outputs/reports/chip_range_multiseed_aggregate_v1.json`
- `outputs/reports/chip_range_multiseed_aggregate_ref1600_v1.json`
- `outputs/reports/chip_range_comparison_seed20.json`
- `outputs/reports/chip_range_comparison_seed21.json`
- `outputs/reports/chip_range_comparison_seed19_refine1600.json`
- `outputs/reports/chip_range_comparison_seed20_refine1600.json`
- `outputs/reports/chip_range_comparison_seed21_refine1600.json`
- `outputs/reports/chip_range_ensemble_v1.json`
- `outputs/reports/chip_range_official100_ref1600_v1.json`
