# CVPR Oracle-Protocol Pre-Submission Review v2

Status date: 2026-11-02

Review target:

- `docs/fp_oracle_control_protocol_status_v1.md`
- `paper/cvpr_manuscript_skeleton_v1.md`
- `paper/cvpr_claim_sections_v3.md`
- `paper/cvpr_experiment_structure_v1.md`
- `paper/tables/residential_sota_main_v5.md`

## Recommendation

Promising protocol-and-evidence submission, conditional on converting the
current notes into manuscript prose and making the oracle-control boundary
visible on the first page.

The paper has a defensible empirical headline because Delta-IoU and Delta-MSE
numerically exceed the published HouseMind-E anchors on every operation while
outside exact preservation and FP-Eval validity remain perfect. It does not need
complete-method superiority to justify a scoped CVPR contribution.

## Strongest Evidence

| Evidence | Why it matters |
|---|---|
| Delta-IoU `0.956703-0.999801` | Broad positive result across five typed operations |
| Delta-MSE `0.000019-0.000096` | All operations are below the public HouseMind-E anchor `0.0019` |
| Outside layers exact `1.0` | Mechanical locality evidence, not a proxy |
| FP-Eval validity `1.0` | Separates fidelity from legality |
| Matched Raster2Seq and editing-aware controls | Rules out a wrapper-only explanation |
| Five seeds, strata, negative controls, attacks | Supports stability and evaluator sensitivity |

## Likely Reviewer Objections

| Objection | Required response |
|---|---|
| The controls are target-derived | State this in the abstract and main protocol section; never call the result deployable |
| HouseMind is not run on the same data | Present it as metric context, not a leaderboard ranking |
| Node/Edge/GED are below HouseMind-E | Report the aligned graph row explicitly and describe it as a limitation |
| Delta-IoU may be dominated by local composition | Include the outside-frozen editing-aware control and its lower result |
| Why only five operation families? | Give the exact scope and list move-wall, resize, free text, and global recomposition as untested |
| FEVP validity differs from FP-Eval validity | Keep separate columns and define both |

## Claim Discipline

Use:

> FloorFlow establishes a strong edit-detection and strict-locality anchor under
> a declared oracle-control protocol, numerically exceeding published
> HouseMind-E Delta-IoU and Delta-MSE anchors while also reporting exact outside
> preservation and legality.

Do not use:

- complete-method SOTA;
- HouseMind leaderboard SOTA;
- uniform SOTA across fidelity, graphs, generation, and editing;
- deployable predicted- or free-text editing;
- "first" without a completed priority search.

## Manuscript Repair List

1. Replace engineering artifact references in the paper body with method and
   protocol prose.
2. Add one figure that explains how the edit region, controls, local delta, and
   audits interact.
3. Put the main per-operation table before any structural appendix.
4. Add qualitative before/after/outside-difference panels for each operation.
5. Move HouseMind comparison into a visually separate context table.
6. Report FP-Eval and independent FEVP validity without collapsing them into one
   score.
7. Add an explicit limitations paragraph about target-derived controls, graph
   metrics, and missing operation families.
8. Avoid version numbers, TODO markers, output paths, and internal project
   identifiers in the final manuscript.

## Submission Readiness

Current state: strong evidence package, incomplete manuscript.

The next blocking work is writing and artifact cleanup, not finding another
headline metric. A concise CVPR story can be built around auditable local
editing, a strong dual-metric oracle-control anchor, and the remaining topology
and deployment boundaries.
