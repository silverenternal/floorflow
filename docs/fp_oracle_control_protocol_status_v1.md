# Oracle-Control Floorplan Editing Protocol and Current Evidence

Status date: 2026-11-02

This is the canonical current summary for the residential oracle-control editing
track. It supersedes stale aggregate ranges in older versioned notes. Historical
`v1` and `v2` artifacts remain useful provenance, but their headline numbers are
not the current submission table.

## Decision

The protocol supports a scoped CVPR contribution:

- an explicit locality and edit-verification contract;
- a five-operation, five-seed oracle-control testbed;
- strong edit-detection metrics under the HouseMind Delta-IoU and Delta-MSE
  definitions;
- exact outside preservation, legality, structural diagnostics, negative
  controls, and protocol attacks;
- paired comparisons against Raster2Seq and an outside-frozen editing-aware
  control.

The intended claim is not complete-method SOTA. The defensible claim is that,
under a declared oracle-control protocol, FloorFlow establishes a strong
fine-metric editing anchor while exposing the locality evidence that external
editing papers do not report.

Oracle controls are target-derived and remain non-deployable. This restriction
does not erase the positive result. It defines its unit of evidence and separates
it from predicted-control and free-text claims.

## Protocol

An evaluation row contains:

1. a pre-edit floorplan;
2. a declared normalized edit region `[x, y, w, h]`;
3. one typed operation: room type, room add, room remove, opening delete, or
   furniture delete;
4. target-derived pre/post coarse controls;
5. a generated post-edit floorplan.

Only the declared edit region may change. The changed-control mask anchors the
edit but cannot enlarge the allowed region. Outside pixels, room labels, room
polygons, walls, openings, and unaffected edges are compared with zero tolerance.

The main source contract is:

- [fp_control_contract_v1.json](../configs/fp_control_contract_v1.json)
- [fp_strict_locality_protocol_v2.md](fp_strict_locality_protocol_v2.md)
- [fevp_formal_spec_v3.md](fevp_formal_spec_v3.md)
- [sota_benchmark_protocol_v2.json](../configs/sota_benchmark_protocol_v2.json)

## Current Main Metrics

CubiCasa test503, 2,479 paired edits, five repeated seeds:

| Operation | Cases | Delta-IoU | Delta-MSE | Outside exact | FP-Eval validity |
|---|---:|---:|---:|---:|---:|
| room type | 487 | 0.999199 | 0.000069 | 1.000000 | 1.000000 |
| room add | 495 | 0.999801 | 0.000030 | 1.000000 | 1.000000 |
| room remove | 500 | 0.999756 | 0.000019 | 1.000000 | 1.000000 |
| opening delete | 502 | 0.956703 | 0.000096 | 1.000000 | 1.000000 |
| furniture delete | 495 | 0.987812 | 0.000083 | 1.000000 | 1.000000 |

The stricter independent FEVP five-check validity ranges from `0.9124` to
`0.9569`. Conditional watertightness ranges from `0.9579` to `0.9675`. These are
reported separately from FP-Eval validity and are not relabeled as `1.0`.

Exact outside pixel, room, polygon, wall, opening, and unaffected-edge
preservation is `1.0` for every operation.

The current per-seed headline values are identical because the evaluated
local-delta path is deterministic on fixed rows. The five repeated seeds are
reproducibility evidence, not an estimate of population-level seed variance.

## Protocol-Aligned Context Comparison

HouseMind-E publishes Delta-IoU `0.608` and Delta-MSE `0.0019` on its RPLAN edit
benchmark. FloorFlow's CubiCasa oracle-control results numerically exceed both
public anchors for every operation. The metric definitions are aligned, while
the datasets and input contracts differ.

| Metric | FloorFlow current range | HouseMind-E public anchor | Conclusion |
|---|---:|---:|---|
| Delta-IoU, higher is better | 0.956703-0.999801 | 0.608 | Every operation exceeds the public anchor |
| Delta-MSE, lower is better | 0.000019-0.000096 | 0.0019 | Every operation beats the public anchor |

The aligned structural readout does not exceed HouseMind-E:

| Metric | FloorFlow aligned | HouseMind-E | Conclusion |
|---|---:|---:|---|
| Node F1 | 0.982250 | 0.998 | Lower |
| Edge overlap | 0.929475 | 0.934 | Lower |
| GED | 0.869544 | 0.467 | Higher and worse |

Thus the paper should claim strong edit-detection and locality evidence, not a
uniform structural or complete-method victory.

## Matched Local Comparisons

Raster2Seq is a reconstruction method, not an editing-aware model. Under an
outside-frozen local-redraw wrapper it reaches Delta-IoU `0.035690-0.164930` by
operation. The editing-aware control baseline reaches
`0.731151-0.998508`. FloorFlow reaches `0.956703-0.999801`.

These comparisons support the claim that the gain is not explained solely by
freezing pixels outside a region.

## Allowed Claim

> On CubiCasa5K test503 with declared oracle controls, FloorFlow reaches
> Delta-IoU 0.956703-0.999801 and Delta-MSE 0.000019-0.000096 across five
> operations and five repeated seeds. These values numerically exceed the
> published HouseMind-E metric anchors of 0.608 and 0.0019 under aligned
> edit-detection definitions. FloorFlow simultaneously preserves outside
> pixel, room, polygon, wall, opening, and edge content exactly, while FP-Eval
> validity is 1.0 and conditional watertightness is 0.9579-0.9675. The result is
> a strong oracle-control protocol anchor, not a matched HouseMind leaderboard
> result or a deployable free-text system.

## Forbidden Claim

- FloorFlow beats the complete HouseMind-E method.
- FloorFlow is leaderboard SOTA on HouseMind's RPLAN benchmark.
- Oracle-control metrics are deployable predicted-control metrics.
- The result establishes free-text editing, move-wall, or resize performance.
- The aligned Node/Edge/GED values exceed HouseMind-E.

## CVPR Framing

The paper should lead with the evaluation problem:

1. Edit fidelity is commonly reported without a mechanical locality contract.
2. Outside preservation, topology legality, and edit fidelity are separate axes.
3. A declared control and region interface makes those axes jointly auditable.
4. FloorFlow provides strong evidence on all axes under one frozen protocol.
5. Protocol attacks and negative controls show that the audit is sensitive.

The empirical headline is the per-operation dual-metric result. The structural
result is a secondary topology diagnostic. Raster2Seq supplies a matched local
comparison, while HouseMind-E supplies metric context rather than a ranking.

## Remaining Boundaries

- Target-derived controls prevent a deployable predicted-control claim.
- HouseMind official inference remains unavailable.
- Free-text, sketch, move-wall, and resize edits lack matched numeric evidence.
- The graph readout remains below the published HouseMind-E Node/Edge/GED
  values.
- Five seeds are stability evidence, not population-level inference.

## Canonical Sources

- [oracle_control_cvpr_dossier_v1.md](../paper/tables/oracle_control_cvpr_dossier_v1.md)
- [residential_sota_main_v5.md](../paper/tables/residential_sota_main_v5.md)
- [residential_sota_claim_gate_v5.json](../outputs/reports/residential_sota_claim_gate_v5.json)
- [residential_edit_test503_v1.json](../outputs/reports/residential_edit_test503_v1.json)
- [residential_editing_aware_baseline_v1.json](../outputs/reports/residential_editing_aware_baseline_v1.json)
- [residential_structural_hardened_v2.json](../outputs/reports/residential_structural_hardened_v2.json)
