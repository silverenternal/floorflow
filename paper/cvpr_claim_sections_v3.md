# CVPR Claim Sections v4

Status date: 2026-11-02. Claim scope: frozen scoped release.

## Abstract

Floorplan editors often report edit accuracy without establishing that unrelated
architecture remains unchanged. We study reliable local execution under an
explicit control interface: an old CAD state, a declared edit region, and a typed
control change. Under target-derived oracle controls, FloorFlow reaches Delta-IoU
`0.956703-0.999801` and Delta-MSE `0.000019-0.000096` across five operations and
2,479 edits while preserving outside layers exactly. Under target-free predicted
controls, it preserves the same outside contract, reaches plan plausibility
`0.986558`, executes all 438 supported development-exposed prompts, and abstains
on all 300 unsupported or ambiguous prompts. We keep these safety properties
separate from fidelity: predicted Room macro IoU is `0.269701`, raw
watertightness is `0.816701`, and an older val191 semantic diagnostic ranges from
`0.00` to `0.92`. The contribution is a scoped, auditable boundary between
locality, legality, and semantic realization, not complete benchmark
superiority.

## Introduction Contributions

1. **Protocol.** A typed-edit contract defines the old state, edit region, coarse
   control change, outside domain, abstention conditions, and verification axes.
2. **Scoped oracle anchor.** Five operations, 2,479 edits, five repeated seeds,
   exact outside preservation, and per-operation edit-support metrics are bound
   to a frozen source registry.
3. **Target-free reliability evidence.** Five val491 seeds establish structural
   plausibility and legality without target controls; a separate
   development-exposed audit establishes supported execution and safe abstention.
4. **Matched diagnostics.** An outside-frozen Raster2Seq wrapper, an editing-aware
   control, deterministic composition arms, and protocol attacks test whether
   locality alone explains the result.
5. **Reproducible release.** Tables, figures, case provenance, source hashes,
   package checksums, and numerical claims share one frozen identity set.

## Results

### Oracle Controls

Per-operation Delta-IoU is `0.999199`, `0.999801`, `0.999756`, `0.956703`, and
`0.987812` for room type, room addition, room removal, opening deletion, and
furniture deletion. The corresponding Delta-MSE values are `0.000069`,
`0.000030`, `0.000019`, `0.000096`, and `0.000083`. Outside pixel, room, polygon,
wall, opening, and unaffected-edge checks are exact. FP-Eval validity is `1.0`;
pooled conditional watertightness is `0.957875-0.967491`. The seed19 trace audit
recomputes all 2,479 rows with zero IoU error.

HouseMind-E publishes Delta-IoU `0.608` and Delta-MSE `0.0019` on RPLAN
natural-language editing. The oracle numbers numerically exceed those anchors
under aligned metric definitions, but the datasets and input contracts differ.
This is therefore a scoped anchor, not a cross-dataset ranking.

### Matched Controls

The outside-frozen Raster2Seq wrapper reaches `0.137160-0.189491`. The matched
editing-aware control reaches `0.731151-0.998508`. FloorFlow reaches
`0.956703-0.999801` on the same operation rows. Paired floorplan-level intervals
and five-operation Holm adjustment are retained in the source report.

The typed composition diagnostic reports Delta-IoU `0.937473`, Node F1
`0.955068`, edge F1 `0.890007`, and GED `2.120855` on 12,395 strict-locality
rows. Relative to the matched no-graph arm, typed routing gains `0.048656` in
Delta-IoU and `-0.061396` in GED. Routing latency is `3.1925 ms` versus
`0.2666 ms`; efficiency is not claimed.

### Predicted Controls

On val491, raster plausibility is `0.984929`, plan plausibility is `0.986558`,
room coverage is `1.0`, room-face retention is `0.995655`, validity is `1.0`, and
outside footprint and room overlap are `0.0`. Raw watertightness is `0.816701`;
functional componentwise watertightness is `0.997963`. Of 491 footprints, 401 are
single-component and pass both checks; 90 multi-component footprints define the
declared protocol boundary.

The operation audit contains 738 development-exposed test prompts: 438 supported
prompts execute and 300 unsupported or ambiguous prompts abstain. Every row has
outside pixel exactness `1.0`, outside room-label exactness `1.0`, and validity
`1.0`. These are safety and state outcomes, not semantic success.

### Semantic Boundary

The older val191 diagnostic reports room addition `92/100`, room type `55/100`,
room removal `20/100`, opening deletion `30/59`, and furniture deletion `0/15`.
It uses a different model age and split from the selected predicted-control
evidence. Predicted room macro IoU is `0.269701`, Node F1 is `0.660202`, edge
overlap is `0.264622`, and GED is `11.600815`.

## Related-Work Positioning

HouseMind unifies floorplan understanding, generation, and editing with discrete
room-instance tokens. Its published table does not expose strict outside
preservation, abstention, attack sensitivity, or paired locality statistics.
Raster2Seq is used as an outside-frozen reconstruction wrapper, not as a native
editing model. The HouseMind official serialization is unavailable, so no
complete-method reproduction is claimed.

## Reviewer-Facing Claim

> Under a frozen typed-edit protocol, FloorFlow provides strong oracle-control
> edit support with exact outside preservation and legality, and a target-free
> predicted-control path with structural plausibility and calibrated abstention.
> Reference fidelity and semantic realization remain explicit limitations.

## Locked Boundaries

- No complete HouseMind, RPLAN, or full-CAD SOTA.
- No uniform Node/Edge/GED superiority.
- No deployable free-text editing.
- No pooling across oracle, predicted, identity, generation, or free-text rows.
- No conversion of abstention, contract checks, or no-op behavior into semantic
  success.
- No fresh held-out claim; the optional extension is separately marked as not
  evaluated.
