# M47 Dihedral Augmentation and NFE1 Budget Probe

Date: 2026-10-31

## M47 Screen

M47 warm-started canonical M27 for 2,000 steps on the frozen 5k subset with the
existing dihedral geometry augmentation. The dev curve total improved slightly
from `0.017198` to `0.017107`, but the legalized endpoint regressed:

| Metric | Relative change vs M36 control |
|---|---:|
| Cost | +7.36% |
| V_rel | +58.67% |
| Grouping | +100.00% |
| b2b HPWL | +68.03% |
| p2b HPWL | +89.89x |
| Boundary | +30.95% |

Cost, V_rel and grouping paired intervals exclude zero in the harmful direction.
The augmentation therefore improves the proxy training distribution while making
the generated layouts less compatible with the downstream refiner/legalizer.
Close augmentation selection for this model; do not sweep subsets after seeing
the legalized failure.

## NFE1/Refiner Curve

Canonical M27 on the frozen legalized dev32 produced:

| Refiner steps | Cost | V_rel |
|---:|---:|---:|
| 400 | 1.16866 | 0.043995 |
| 800 | 1.15349 | 0.040825 |
| 1600 | 1.11336 | 0.031058 |

This is a positive within-model budget curve, but it is not a cross-model win.
On the frozen Official-8 cohort with identical NFE1/ref1600 settings:

| Model | Cost | V_rel |
|---|---:|---:|
| Canonical M27 | 1.10152 | 0.03537 |
| Trinity d192 | 1.03978 | 0.01455 |

The candidate loses cost by `+5.94%` and V_rel by `+143.10%`; both paired
intervals exclude zero in the harmful direction. M36 control and NFE4 variants
also fail to beat the exact anchor. Small-block case index 0 remains a shared
failure: raw b2b/area are near target, but legalization expands the b2b and area
gaps to roughly `+8–10%`.

## Next Constraint

Do not claim a refiner-budget or checkpoint-routing win. The next investigation
must target raw-to-legalized displacement and boundary compatibility. Increasing
hard-projector iterations alone is not accepted: on the diagnostic 21-block case,
the current pairwise projector needs roughly 180 iterations to clear residual
overlaps, and the over-relaxed variant that clears overlaps faster raises
boundary violations across dev32.

## Artifacts

- `configs/chip_m47_dihedral_augmentation_v1.json`
- `outputs/reports/chip_m47_dihedral_quick_refiner400_dev32.json`
- `outputs/reports/chip_m27_canonical_nfe1_refiner_curve_v1.json`
- `outputs/reports/chip_m27_canonical_nfe1_ref1600_official8_v1.json`
- `outputs/reports/chip_trinity_d192_nfe1_ref1600_official8_v1.json`
