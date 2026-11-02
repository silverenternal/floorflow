# E1 External Routed-PPA Decision

Date: 2026-11-01

## Decision

The frozen external cohort is complete and DRC-clean, but routed PPA does not
transfer:

| Quantity | Result |
|---|---:|
| Designs | 8 |
| Completed candidate/control pairs | 18 |
| Detail-route DRC | 0 on both arms |
| Routed wirelength win / loss | 11 / 7 |
| Routed wirelength mean change | +0.656% |
| Median routed wirelength change | -0.874% |
| Congestion mean change | +2.466% |
| Timing WNS mean change | +0.00470 ns |
| Power worst regression | +4.05% |
| Standard-cell area worst regression | +2.38% |

The mean routed wirelength and congestion endpoints fail promotion. Do not
claim external routed-PPA improvement, external SOTA or surrogate-to-PPA
transfer.

## Why This Is the Primary External Result

The 8-design/18-pair cohort has:

- a manifest frozen before these outcomes;
- complete candidate/control pairs;
- zero detail-route DRC in both arms;
- routed wirelength, congestion, timing, power and area metrics;
- identical pinned ORFS image and pair protocol.

The or1200 pin-alignment sensitivity run has 1,944 DRC and missing hold metrics.
It is retained as negative routeability evidence, not as a primary PPA result.
Further route-iteration sweeps are not justified by the observed plateau.

## Paper Boundary

The defensible paper structure is:

1. internal absolute-budget structural and primary-metric result;
2. explicit unresolved area gap;
3. frozen external negative transfer result;
4. no SOTA or PPA-transfer claim.

This is a publishable mechanism/evidence package only if the paper is framed as
structured generation and rigorous transfer analysis, not as an SOTA placement
paper.

## Artifacts

- `scripts/build_chip_e1_external_audit.py`
- `outputs/reports/chip_e1_external_holdout_v1.json`
- `outputs/reports/chip_external_ppa_holdout_v44_all.json`
- `outputs/reports/orfs/chipbench_m49_p32_pinalign_or1200_v2.json`
