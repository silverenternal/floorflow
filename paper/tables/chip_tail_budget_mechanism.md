# Chip Tail and Equal-Quality Budget Mechanism

Scope: matched FloorSet-Lite official-100 diagnostics. The primary endpoint remains NFE4 plus 400 refiner steps.

| Metric | Quantile | Baseline | Candidate | Improvement | Paired 95% CI |
|---|---:|---:|---:|---:|---:|
| cost | p50 | 1.16449 | 1.13434 | +2.59% | [+0.19%, +5.92%] |
| cost | p75 | 1.34445 | 1.25744 | +6.47% | [+0.07%, +20.13%] |
| cost | p90 | 1.85437 | 1.58616 | +14.46% | [+1.20%, +27.87%] |
| cost | p95 | 2.15724 | 1.84683 | +14.39% | [-39.57%, +27.19%] |
| v rel | p50 | 0.04396 | 0.03779 | +14.04% | [-15.64%, +29.93%] |
| v rel | p75 | 0.08203 | 0.06709 | +18.21% | [-1.74%, +33.71%] |
| v rel | p90 | 0.13477 | 0.11384 | +15.53% | [-6.62%, +30.61%] |
| v rel | p95 | 0.16423 | 0.15641 | +4.76% | [-19.65%, +33.08%] |

The cost gain rises from p50 to p90/p95. V_rel improves most near p75-p90 but is not monotonic at p95.

## Structural-Failure Concentration

| Metric | Baseline-cost stratum | Cases | Improvement | Paired 95% CI |
|---|---|---:|---:|---:|
| cost | Top 25% | 25 | +9.99% | [-1.49%, +20.89%] |
| cost | Remaining 75% | 75 | +0.88% | [-0.92%, +2.54%] |
| v rel | Top 25% | 25 | +18.24% | [+2.88%, +32.93%] |
| v rel | Remaining 75% | 75 | +3.83% | [-11.43%, +16.91%] |

## Equal-Quality Budget

| Setting | Candidate | Baseline | Saved | Boundary |
|---|---:|---:|---:|---|
| NFE | 1 | 8 | 7 | Pooled quality threshold |
| Refiner steps | 100 | 400 | 300 | Seed-19 controlled curve |

At 50 refiner steps, cost is 1.31952 against threshold 1.36409, but V_rel is 0.06159 against threshold 0.05961. The short-budget V_rel deficit is therefore retained.

Allowed claim: the candidate primarily reduces structural-failure tails and can reach declared pooled quality with fewer inference steps. Do not claim uniform case-level improvement, monotonic extreme-tail advantage, or replacement of the primary endpoint.

Figure: `paper/figures/chip_tail_budget_mechanism.svg`.
