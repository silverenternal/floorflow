# M43-M44 Projector-Aware Endpoint Contract

Date: 2026-10-31

## Motivation

M42 showed that internal hierarchy consistency can produce less legalizable
geometry. M43 instead exposed the deployment hard-projector value in the
training endpoint while using a straight-through gradient to the raw x0.

## Full Strength

Strength `1.0` diverged:

- total loss `11.23`;
- latent MSE `32.38`;
- hierarchy bbox loss `38.90`;
- endpoint displacement `15.60`.

Full projected values lie outside the raw-x0 trust region, so M43 is rejected.

## Bounded Strength

M44 used one predeclared trust-region strength of `0.1`. Training remained
stable, but the legalized dev32 endpoint regressed:

| Metric | Relative |
|---|---:|
| Cost | +3.67% |
| V_rel | +22.73% |
| Area gap | +16.60% |
| b2b HPWL gap | +20.88% |
| p2b HPWL gap | +12.56x |
| Grouping | +13.64% |
| Boundary | +26.19% |

V_rel and boundary paired intervals exclude zero in the harmful direction.

## Decision

Close endpoint-value substitution and do not sweep projector strength.

M45 will instead keep the M27 x0 unchanged and learn a separate
projector-displacement output head supervised by detached projector deltas.
This preserves the base prediction while giving legalization a dedicated model
pathway.

## Artifacts

- `src/floorflow/chip_flow/projector_endpoint_contract.py`
- `tests/test_chip_projector_endpoint_contract.py`
- `configs/chip_m43_projector_endpoint_contract_v1.json`
- `configs/chip_m44_trust_region_projector_endpoint_v1.json`
- `outputs/reports/chip_m44_projector_endpoint01_quick_refiner400_dev32.json`
