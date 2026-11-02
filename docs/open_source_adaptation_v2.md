# Open-source adaptation map (2026-10-25)

The project follows a reuse-first rule: a local algorithm is written only when
none of the pinned implementations below exposes the required primitive.

| Capability | Pinned source | Adapted module | What changed for floorplans |
|---|---|---|---|
| Dynamic query and mask assignment | `third_party/EoMT` @ `aa017c3` (MIT) | `floorflow.model.learned_mask_generator` | Room/object queries consume contour features; Hungarian assignment is class-aware. |
| Sparse route attention | `third_party/MiTA` @ `95460f7` (MIT) | `floorflow.model.official_mita_route` | Per-router top-k keys are shared by semantic families and recursive room nodes. |
| Exact-total dynamic token budgets | `third_party/DHSA` @ `069d553` (Apache-2.0) | `floorflow.model.dynamic_budget` | Budgets are family masks rather than attention rows and preserve exact totals. |
| Dynamic masked partial layouts | `third_party/MaskPLAN` @ `4af728a` (MIT) | data/control masks in the distribution training path | Coarse controls are training-only perturbations; inference remains contour conditioned. |
| Prompt-mask query decoding | `third_party/PMT` @ `0e80372` (MIT) | `floorflow.model.pmt_mask_decoder` | Prompts are room semantics, wall context and predicted spatial support. |
| Room token denoising and transport | `third_party/Trinity` @ `51334c5` (MIT) | `floorflow.model.trinity_contour_room` | Molecular/set tokens become contour room tokens; output is a CAD raster. |
| Staged layout refinement | `third_party/house-diffusion` @ `083c9c5` (GPL, conceptual only) | `floorflow.recursion.inward_flow` | Only the coarse-to-fine scheduling idea is used; no GPL source is copied. |
| Wall/room vector evaluation | `third_party/fpvec-lab` @ `b4b4a16` (MIT) | `floorflow.evaluation.fpeval_bridge` | MIT evaluation implementation is imported and fed project taxonomies. |

## New recursive graph adaptation

`floorflow.recursion.graph_adapter` uses the pinned MiTA route attention
directly. Each node is a contour-derived partition region with semantic logit
statistics and parent/depth/area descriptors. Parent and sibling evidence is
routed before a zero-initialized residual is applied to room logits. The zero
initialization makes the adapter an exact identity for existing checkpoints.

The final room composition uses
`floorflow.inference.gated_room_decoupling`: graph confidence gates geometry,
while each graph component receives a semantic class from the base semantic
head. This is an explicit instance-geometry / room-semantics decomposition
rather than a global on/off switch.

## Scope and leakage controls

- No external person is used for annotation, rating or selection.
- Target labels are used only in the training loss and validation selection.
- Two contracts are reported separately. The target-free distribution path
  never receives a target-derived coarse layout. The layout-conditioned
  completion path receives a serialized coarse control and is an oracle
  upper-bound diagnostic when that control is derived from the test target.
- Main deployable results use independently predicted or explicitly external
  controls. Oracle controls never enter a deployable main table.
- Main results are single-sample outputs, not best-of-N.
