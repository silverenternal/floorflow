# Evidence-Aligned CVPR Claim Sections

Status: historical. The current submission-facing claim language is
`cvpr_claim_sections_v3.md`; the protocol authority is
`docs/fp_oracle_control_protocol_status_v1.md`.

This file contains canonical abstract, introduction, and related-work claim
language for internal review. It is not a complete manuscript.

## Abstract

Floorplan editing systems increasingly report edit accuracy, but accuracy alone
does not establish whether an edit is actually local. We introduce the FloorPlan
Editing Verification Protocol (FEVP), an attack-tested audit layer for pixel,
room-instance, polygon, edge, graph, topology, and edit-region behavior. FEVP
separates strict outside preservation from editing fidelity and reports missing
external evidence as unreported rather than failed. Under the same protocol,
FloorFlow's control-gated local-delta pipeline achieves outside pixel, room,
label, polygon, and edge exactness of 1.0 over 12,395 five-seed rows, with
Locality Violation Rate 0.0 and Delta-IoU 0.9871 (95% Student-t seed CI
[0.9845, 0.9897]). Removing edit gating lowers room-level exactness to 0.4863
and raises violation rate to 0.0139. The protocol is portable to 200 CE2EPlan
generation rows and exposes deterministic sensitivity to controlled corruption.
We do not claim that FloorFlow beats HouseMind or that unreported audit cells
show inferiority; FEVP makes a previously implicit editing requirement
independently auditable.

## Introduction Contributions

1. **Auditability problem.** We identify that edit-quality metrics do not by
   themselves establish whether an edit obeys a declared locality contract.
2. **Protocol.** FEVP v1.0-hardened defines strict pixel equality, one-to-one
   room matching, polygon and typed-edge preservation, continuous violation,
   five-check validity, localized watertightness, graph precision/recall/F1,
   and fixed-evaluator paired GED.
3. **Mechanism.** Typed routing remains on the audited locality-quality Pareto
   frontier and outperforms dense, no-gate, and graphless routing on the fixed
   evaluator's joint quality objectives; compute and parameter boundaries are
   disclosed.
4. **Robustness.** Deterministic protocol attacks achieve zero observed
   false-pass behavior across all mandatory metrics, and 400 conflict-prompt
   proxies retain strict locality in the typed arm.
5. **Portability.** The same evaluator decomposes structural behavior of 200
   CE2EPlan generation rows without using FloorFlow-specific APIs. Outside
   exactness is not applicable to this generation-only target.

The contribution is protocol and evidence architecture, not a new generation
SOTA or a complete CAD generation claim.

## Related Work

### HouseMind

HouseMind introduces discrete room-instance tokens and a unified vocabulary to
connect floorplan geometry with symbolic reasoning. Its multimodal alignment and
instruction tuning cover understanding, generation, and editing within one MLLM
framework [HouseMind, CVPR 2026]. These are substantive contributions to
multimodal floorplan reasoning.

HouseMind-E and HouseMind-O report edit-level Delta-IoU, Delta-MSE, Node F1,
edge overlap, and GED. Their public evaluation does not report outside exactness,
violation rate, single-room negative controls, FEVP attacks, or paired locality
statistics. This absence defines an audit boundary, not a performance ranking.
Official HouseMind inference code is marked Code (Soon), so this work does not
claim a matched comparison or a defeat.

### CE2EPlan

CE2EPlan performs end-to-end boundary-conditioned vector floorplan generation
and introduces topology- and geometry-enhanced components that remove the need
for a multi-stage intermediate representation [CE2EPlan, IEEE TVCG 2026].
It is a generation method, not an editing-aware baseline.

We use 200 deterministic, evenly spaced public CE2EPlan generation outputs only
to test evaluator portability. FEVP decomposes their validity, watertightness,
node/edge quality, and GED. Outside exactness and violation rate are marked N/A
because no before/after edit pair exists.

### Raster2Seq

Raster2Seq reconstructs vector floorplans from rasterized inputs and is used
here only as a matched reconstruction baseline. The edit-row sensitivity study
uses shared project preprocessing, not Raster2Seq's official preprocessing, and
Raster2Seq is not an editing-aware model. Its outside-frozen local-redraw
wrapper is therefore a diagnostic control, not a claim about native editing
capability.

### Positioning

HouseMind and CE2EPlan advance complementary directions: unified multimodal
floorplan reasoning and end-to-end generation. FEVP addresses the orthogonal
question of whether edit locality and topology can be audited, decomposed, and
attacked under a shared protocol. The contribution is intended to be compatible
with, not substitutive for, stronger generators or unified multimodal systems.

## Boundaries

- No HouseMind or CE2EPlan leaderboard ranking.
- No inference that unreported evidence is failure.
- No generation result is interpreted as an editing result.
- No watertight-guard net-benefit claim.
- No claim of robustness to arbitrary free-form natural language.
- No human-preference claim.
