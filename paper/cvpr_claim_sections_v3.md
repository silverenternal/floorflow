# CVPR Claim Sections v3

Status date: 2026-11-02

This file is the current submission-facing claim language. It supersedes the
aggregate ranges in `cvpr_claim_sections_v1.md`. The protocol and evidence
authority is `docs/fp_oracle_control_protocol_status_v1.md`.

## Abstract

Floorplan editing systems increasingly report edit accuracy, but accuracy alone
does not show whether an edit leaves unrelated geometry unchanged. We study
strictly local floorplan editing under an explicit control protocol that fixes
the pre-edit plan, edit region, typed operation, and available controls. We
evaluate five edit families on CubiCasa5K test503 with five repeated seeds:
room-type change, room addition, room removal, opening deletion, and furniture
deletion. FloorFlow reaches Delta-IoU 0.956703-0.999801 and Delta-MSE
0.000019-0.000096 under the HouseMind edit-detection definitions. These values
exceed the published HouseMind-E anchors 0.608 and 0.0019 while using different
datasets and input contracts. FloorFlow preserves outside pixels, room labels,
polygons, walls, openings, and unaffected edges exactly, maintains FP-Eval
validity 1.0, and reports conditional watertightness 0.9579-0.9675. An
attack-tested FEVP evaluator and matched Raster2Seq and editing-aware controls
separate locality from edit fidelity. The result is a scoped oracle-control
anchor, not a complete-method HouseMind ranking or a deployable free-text claim.

## Introduction Contributions

1. **Protocol.** We define a floorplan editing contract that makes the edit
   region, typed operation, coarse controls, outside domain, and verification
   metrics explicit.
2. **Auditability.** We separate edit fidelity from exact outside preservation,
   topology legality, structural consistency, and edit-region behavior.
3. **Strong scoped evidence.** Across five operations and five seeds, FloorFlow
   numerically exceeds the published HouseMind-E Delta-IoU and Delta-MSE
   anchors while preserving exact outside content and validity.
4. **Mechanism.** Control-gated local composition and typed routing improve edit
   fidelity without converting the operation into a global reconstruction.
5. **Matched controls.** Raster2Seq and an outside-frozen editing-aware wrapper
   are evaluated on shared paired rows and the same scorer, ruling out a
   locality-wrapper explanation for the entire gain.
6. **Boundary evidence.** Negative controls, protocol attacks, complexity strata,
   independent FEVP validity, and graph diagnostics expose where the protocol
   succeeds and where topology remains weak.

## Results

Under the oracle-control protocol, FloorFlow's per-operation Delta-IoU is
0.999199 for room-type change, 0.999801 for room addition, 0.999756 for room
removal, 0.956703 for opening deletion, and 0.987812 for furniture deletion.
The corresponding Delta-MSE values are 0.000069, 0.000030, 0.000019, 0.000096,
and 0.000083. Every operation preserves pixel, room, polygon, wall, opening,
and unaffected-edge content exactly. FP-Eval validity is 1.0 and conditional
watertightness ranges from 0.9579 to 0.9675. The primary local-delta path is
deterministic on fixed rows, so the repeated seeds verify reproducibility rather
than estimate seed-level sampling variance.

HouseMind-E publishes Delta-IoU 0.608 and Delta-MSE 0.0019. FloorFlow numerically
exceeds both anchors for all five operations under aligned metric definitions.
Because HouseMind reports on RPLAN natural-language editing and FloorFlow reports
on CubiCasa oracle controls, this is a protocol-aligned comparison rather than a
same-benchmark rank.

The matched outside-frozen Raster2Seq wrapper reaches Delta-IoU
0.035690-0.164930. The editing-aware control baseline reaches
0.731151-0.998508. FloorFlow's range is 0.956703-0.999801. Exact outside
preservation remains 1.0 in every FloorFlow operation.

The aligned structural readout reaches Node F1 0.982250, edge overlap 0.929475,
and GED 0.869544. These values do not exceed HouseMind-E's 0.998, 0.934, and
0.467. The graph result is therefore a secondary topology diagnostic, not an
additional SOTA claim.

## Related Work Positioning

HouseMind introduces discrete room-instance tokens and a unified multimodal
interface for floorplan understanding, generation, and editing. Its edit table
reports Delta-IoU, Delta-MSE, Node F1, edge overlap, and GED but does not report
strict outside exactness, locality violation rate, negative controls, protocol
attacks, or paired locality statistics. This absence defines an audit boundary,
not a failure claim.

Raster2Seq reconstructs vector plans from raster inputs and is used here only as
a matched reconstruction baseline. Its outside-frozen local-redraw wrapper is an
editing-aware diagnostic rather than a claim about native editing capability.

CE2EPlan, TLC-Plan, and FMLM remain generation-only context. No generation
result is interpreted as an editing result.

## Reviewer-Facing Claim

> Under one frozen oracle-control editing protocol, FloorFlow produces
> edit-detection metrics above the published HouseMind-E anchors while also
> reporting exact outside preservation and topology legality. The evidence
> supports a strong scoped editing anchor, not complete-method or cross-benchmark
> SOTA.

## Locked Boundaries

- No claim that FloorFlow beats the complete HouseMind-E method.
- No claim that FloorFlow is leaderboard SOTA on HouseMind's RPLAN benchmark.
- No claim that aligned Node/Edge/GED exceed HouseMind-E.
- No claim that oracle controls are deployable predicted controls.
- No claim of free-text, move-wall, resize, or global-recomposition capability.
- No pooling of aligned, canonical-instance, and typed graph protocols.
- No conversion of missing external metrics into zeros or failures.
