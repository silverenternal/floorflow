# Floorplan Claim-Evidence Matrix v1

Status date: 2026-11-02

| Claim | Authoritative evidence | Evaluation unit | Status | Required caveat |
|---|---|---|---|---|
| Strict outside preservation is auditable | FEVP outside layers; 12,395 rows and 7,345 controls | Case row | Supported | Oracle-control diagnostic |
| Oracle-control edit fidelity is strong | Delta-IoU 0.956703-0.999801; Delta-MSE 0.000019-0.000096 | Five operations, test503, five seeds | Supported | Not a HouseMind leaderboard rank |
| Oracle outputs remain legal | FP-Eval validity 1.0; conditional watertightness 0.9579-0.9675 | Five operations | Supported | FEVP five-check validity is 0.9124-0.9569 |
| Local wrapper alone is insufficient | Raster2Seq wrapper and editing-aware control comparisons | Matched edit rows | Supported | Raster2Seq is reconstruction-only |
| Target-free outputs are structurally plausible | Raster plausible 0.98493; plan plausible 0.98656; validity 1.0 | val491, five seeds | Validation-supported | Raw fpeval watertightness is 0.81670 |
| Unsupported operations are rejected safely | 300/300 abstentions in the latest development exposure | Prompt category | Development-supported | Test manifest exposure is not held-out evidence |
| Supported operations are executed | 438/438 supported prompts executed | Prompt category | Development-supported | Execution is not semantic success |
| Semantic operation success is high | Preliminary val191 audit | Prompt category | Not established | Rates are 0.00-0.92 and fail several categories |
| Predicted reconstruction fidelity is high | Room macro IoU 0.26970 | val491 | Not established | Identity, wall, opening, and graph metrics remain weak |
| Complete HouseMind or RPLAN SOTA | None | N/A | Forbidden | No matched executable evidence |

## Claim Language

Use:

> FloorFlow separates edit reliability from reference fidelity. It provides a
> strong oracle-control anchor, exact outside preservation, and a target-free
> path with high structural plausibility and safe abstention. Semantic operation
> realization remains an explicit boundary.

Do not use:

- complete-method or uniform SOTA;
- deployable free-text editing;
- high predicted-control Room IoU;
- semantic success inferred from exact locality;
- test development exposure as held-out evidence.
