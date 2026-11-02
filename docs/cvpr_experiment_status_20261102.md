# CVPR Experiment Status

Status date: 2026-11-02

This document replaces the 2026-10-26 status as the current submission-facing
experiment summary. The canonical protocol-level evidence is
[fp_oracle_control_protocol_status_v1.md](fp_oracle_control_protocol_status_v1.md).

## Selected Submission Direction

The paper is structured as a floorplan-editing protocol and evidence study. Its
main contribution is an explicit, attack-tested locality contract combined with
a strong oracle-control evaluation. It is not positioned as complete-method SOTA
over HouseMind, RPLAN SOTA, or deployable free-text editing.

The advisor-supported packaging decision is:

> A scoped result is sufficient when the paper clearly identifies which metrics
> exceed a public anchor, which matched baselines are beaten, and which
> deployability or benchmark claims remain outside the evidence.

## Current Headline

On CubiCasa5K test503 with declared oracle controls:

| Metric | Current evidence |
|---|---|
| Delta-IoU | `0.956703-0.999801`, five operations and five seeds |
| Delta-MSE | `0.000019-0.000096` |
| Outside pixel/room/polygon/wall/opening/edge exactness | `1.0` |
| FP-Eval validity | `1.0` |
| Conditional watertightness | `0.9579-0.9675` |
| FEVP five-check validity | `0.9124-0.9569` |
| HouseMind-style Node F1 | `0.982250` |
| HouseMind-style edge overlap | `0.929475` |
| HouseMind-style GED | `0.869544` |

HouseMind-E publishes Delta-IoU `0.608`, Delta-MSE `0.0019`, Node F1 `0.998`,
edge overlap `0.934`, and GED `0.467`. FloorFlow numerically exceeds the first
two anchors but not the graph anchors. The paper must state that distinction.

## Claim Ladder

| Level | Status |
|---|---|
| Strong oracle-control edit-detection anchor | Passed |
| Exact outside preservation with audited validity | Passed |
| Matched Raster2Seq and editing-aware local comparisons | Passed |
| Predicted-control deployment | Open |
| Free-text instruction editing | Not claimed |
| Complete HouseMind/RPLAN/full-CAD SOTA | Blocked |

## Submission Tables

1. [residential_sota_main_v5.md](../paper/tables/residential_sota_main_v5.md):
   per-operation main editing metrics and external context.
2. [residential_graph_protocol_dominance_v1.md](../paper/tables/residential_graph_protocol_dominance_v1.md):
   structural protocol separation.
3. [fp_complexity_strata.md](../paper/tables/fp_complexity_strata.md):
   room-count, interface, and edit-area strata.
4. [fevp_hardened_main_results_v2.md](../paper/tables/fevp_hardened_main_results_v2.md):
   attack-tested FEVP evidence.
5. [fp_limitations_v2.md](../paper/tables/fp_limitations_v2.md):
   reviewer-facing limitations.

## Remaining Work Before Submission

- Convert the current claim and experiment notes into a complete manuscript.
- Replace internal artifact names in the paper body with methodological prose.
- Keep HouseMind-E values in a visually separate context row or table.
- Report the independent FEVP five-check validity separately from FP-Eval.
- Add qualitative panels that show exact outside preservation and the typed
  operation families.
- State the oracle-control boundary in the abstract or first page, not only in
  the appendix.

## Canonical Evidence

- [residential_sota_claim_gate_v5.json](../outputs/reports/residential_sota_claim_gate_v5.json)
- [residential_sota_main_results_v5.json](../outputs/reports/residential_sota_main_results_v5.json)
- [residential_edit_test503_v1.json](../outputs/reports/residential_edit_test503_v1.json)
