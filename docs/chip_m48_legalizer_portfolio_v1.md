# M48 Two-Trajectory Legalizer Portfolio

Date: 2026-10-31

> Post-run correction: the original FloorSet artifacts used a hard projector
> that incorrectly treated fixed-shape blocks as position-immovable. The
> pre-fix results below are retained only for provenance. The post-fix dev32
> diagnostic still improves mean cost (`-3.68%`, CI crosses zero) and V_rel
> (`-27.34%`, CI excludes zero), but the Official-100 claim requires a post-fix
> rerun.

## Finding

The remaining FloorSet gap is dominated by raw-to-legalized displacement. On
dev32, legalization contributes roughly 55% of normalized center displacement.
The refiner improves HPWL and area but more than doubles boundary violations;
legalization then restores V_rel at the cost of b2b and p2b gaps.

This motivated a legalizer portfolio rather than another residual network.

## Candidate

The candidate generates two trajectories from the same checkpoint:

- projector iterations `8`;
- projector iterations `32`.

It legalizes:

1. projector8/restored;
2. projector8/expanded relations;
3. projector32/restored.

Selection is target-free and uses:

```text
V_rel
-> refined-to-legalized displacement
-> boundary
-> grouping
-> candidate name
```

A candidate is ineligible if its legalizer rung or solve count exceeds the
projector8/restored baseline.

## Budget

The result uses two NFE trajectories, two 1,600-step refiners and three
legalizer calls. It is an absolute-budget result and must not be presented as
same-inference or same-compute.

## Official-100, Three Inference Seeds

Exact anchor: Trinity d192, NFE1, refiner1600, group2/boundary2, one sample.

| Seed | Candidate cost | Candidate V_rel | Anchor cost | Anchor V_rel |
|---:|---:|---:|---:|---:|
| 19 | 1.07036 | 0.014955 | 1.10541 | 0.029818 |
| 20 | 1.06713 | 0.016058 | 1.10498 | 0.026394 |
| 21 | 1.06876 | 0.015810 | 1.13191 | 0.033094 |

Pooled 300-case paired result:

| Metric | Relative | 95% CI | Improved / worsened |
|---|---:|---:|---:|
| Cost | -4.07% | [-7.49%, -1.88%] | 151 / 138 |
| V_rel | -47.57% | [-1.83%, -1.02%] absolute | 117 / 38 |

Feasibility is `1.0` for every run, and there are no legalizer rung/solve
escalations.

## Component Boundary

On Official-100 seed19, the candidate significantly improves p2b, boundary and
grouping, while b2b and area worsen. The claim is therefore structural legality
and cost/V_rel under the declared budget, not universal component superiority.

## Evidence

- seed19 portfolio: `outputs/reports/chip_m48_portfolio_official100_v1.json`
- seed20 portfolio: `outputs/reports/chip_m48_portfolio_official100_seed20.json`
- seed21 portfolio: `outputs/reports/chip_m48_portfolio_official100_seed21.json`
- exact anchors: `outputs/reports/chip_trinity_d192_nfe1_ref1600_official100_*.json`
- pooled report: `outputs/reports/chip_m48_portfolio_three_seed_v1.json`
- evaluator: `scripts/evaluate_chip_m48_portfolio.py`
- selector: `src/floorflow/chip_flow/legalizer_portfolio.py`

External routed PPA and same-inference claims remain unsupported.
