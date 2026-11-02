# Strictly Local Floorplan Editing Under an Auditable Control Protocol

## Position

Floorplan editors increasingly report high edit accuracy, but edit accuracy
alone does not show that an operation leaves unrelated geometry unchanged.
This paper treats locality, topology legality, and edit fidelity as separate
measurement axes and evaluates FloorFlow through one explicit protocol.

The paper is a protocol-and-evidence contribution with a strong oracle-control
result. It does not claim complete-method SOTA over HouseMind, RPLAN SOTA,
deployable free-text editing, or uniform superiority on every metric.

## Claim

Given a pre-edit floorplan, a declared edit region, and typed pre/post controls,
FloorFlow generates a post-edit floorplan that changes only the declared region
and remains legal. Under the HouseMind edit-detection metric definitions,
FloorFlow reaches Delta-IoU `0.956703-0.999801` and Delta-MSE
`0.000019-0.000096` across five edit operations and five repeated CubiCasa
test503 seeds. These values exceed the published HouseMind-E anchors `0.608`
and `0.0019`; the datasets and input contracts differ, so this is a scoped
protocol anchor rather than a leaderboard rank.

Outside pixels, room labels, room polygons, walls, openings, and unaffected
edges are preserved exactly. FP-Eval validity is `1.0`; conditional
watertightness is `0.9579-0.9675`; independent FEVP five-check validity is
reported separately as `0.9124-0.9569`.

## Contributions

1. A floorplan editing protocol that separates declared edit support, exact
   outside preservation, topology legality, and edit fidelity.
2. An attack-tested verification layer covering pixels, room instances,
   polygons, walls, openings, edges, graph consistency, and locality violations.
3. A typed oracle-control testbed with room-type, room-add, room-remove,
   opening-delete, and furniture-delete operations.
4. A strong dual-metric editing result under aligned HouseMind Delta-IoU and
   Delta-MSE definitions, together with exact outside preservation and legality.
5. Matched comparisons against a reconstruction wrapper and an outside-frozen
   editing-aware control, showing that the gain is not explained by freezing
   alone.
6. Structural diagnostics and protocol attacks that expose the remaining graph
   and watertightness boundaries.

## Method

- Inputs: pre-edit plan, normalized edit region, typed operation, and pre/post
  coarse controls.
- Completion: room, structure, furniture, and support refiners generate the
  post-edit plan.
- Execution: typed operations edit only the intended control family or room
   partition.
- Composition: a control-change gate composes the post-edit result only inside
   the declared region.
- Verification: outside layers are restored exactly; topology is checked with
   FP-Eval and FEVP.
- Diagnostics: graph quality is reported through unique room-type and canonical
   instance protocols without pooling them.

## Protocol

The valid outside domain is:

```text
O = footprint AND NOT edit_region AND NOT dilate(wall_before OR wall_after, 1)
```

Changes outside `O` are contract failures, not small quality losses. The
changed-control mask is an anchor for the requested edit and cannot expand the
allowed region.

## Main Results

| Operation | Cases | Delta-IoU | Delta-MSE | Outside exact | FP-Eval validity |
|---|---:|---:|---:|---:|---:|
| room type | 487 | 0.999199 | 0.000069 | 1.000000 | 1.000000 |
| room add | 495 | 0.999801 | 0.000030 | 1.000000 | 1.000000 |
| room remove | 500 | 0.999756 | 0.000019 | 1.000000 | 1.000000 |
| opening delete | 502 | 0.956703 | 0.000096 | 1.000000 | 1.000000 |
| furniture delete | 495 | 0.987812 | 0.000083 | 1.000000 | 1.000000 |

The matched local-redraw Raster2Seq wrapper reaches Delta-IoU
`0.035690-0.164930`. The editing-aware control baseline reaches
`0.731151-0.998508`. FloorFlow exceeds both ranges while preserving exact
outside content.

The aligned structural readout is Node F1 `0.982250`, edge overlap `0.929475`,
and GED `0.869544`. It remains below the published HouseMind-E values `0.998`,
`0.934`, and `0.467`; the paper reports this as a topology limitation.

## Experiments

1. Main editing evaluation on CubiCasa test503, 2,479 paired edits, five seeds.
2. Per-operation Delta-IoU, Delta-MSE, exact outside preservation, legality, and
   conditional watertightness.
3. Matched Raster2Seq and editing-aware local comparisons.
4. HouseMind-style and canonical structural protocols, reported separately.
5. Room-count, edit-area, and interface-complexity strata.
6. Single-room negative controls and deterministic outside-preservation attacks.
7. Independent FEVP five-check validity and watertightness diagnostics.

## Figures

1. Protocol diagram: plan, region, typed control, local composition, audits.
2. Main metric figure: five operations against HouseMind anchors and the
   editing-aware control.
3. Locality panel: before/after overlays and exact outside preservation.
4. Structural frontier: Node/Edge/GED by protocol and ablation.
5. Failure taxonomy: weak-interface and low-signal watertightness boundaries.

## Limitations

- Controls are target-derived; this is not deployable predicted-control or
  free-text editing.
- HouseMind official inference is unreleased, so the paper reports metric
  alignment rather than a same-benchmark rank.
- Node/Edge/GED do not exceed the published HouseMind-E anchors.
- Move-wall, resize, and global-recomposition edits lack matched numeric
  evidence.
- Five seeds establish stability, not population-level inference.
- The primary local-delta path is deterministic on fixed rows, so repeated seeds
  verify reproducibility rather than create independent sampling variation.
- Graph GED remains protocol-sensitive; aligned and instance graphs are never
  pooled.

## Reviewer Defense

The empirical claim is deliberately narrower than complete-method SOTA:

> Under one frozen oracle-control protocol, FloorFlow produces stronger
> edit-detection metrics than the published HouseMind-E anchors while also
> providing exact locality and legality evidence that the public comparison does
> not report.

This wording names the evaluated unit, split, metric definitions, matched
comparisons, and limitation. It supports a CVPR paper about verifiable local
editing without overstating dataset or benchmark equivalence.
