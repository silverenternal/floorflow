# FloorFlow Strict-Locality Editing Protocol v2

This document defines the evaluated editing contract. It is not a manuscript.

## 1. Edit-region contract

- The declared normalized rectangle is clamped to `[0,1]` before rasterization.
- The output is allowed to change only inside that region.
- The changed-control mask is an anchor for the declared edit, not an expansion of the allowed region.

## 2. Outside exactness

- `V_out = footprint AND NOT dilate(wall_before OR wall_after,1) AND NOT edit_region`.
- Outside pixel exact uses strict `uint8` equality with no tolerance.
- Outside room-label exact uses strict semantic-label equality.

## 3. Editing quality

- Delta-IoU compares expected and predicted changed pixels over the valid domain.
- Delta-MSE is the binary change-map MSE and is a diagnostic, not the main contribution.
- GED is interpreted together with operation type and its edit-path decomposition.

## 4. Topology legality

- `fpeval.verify` checks floorplan validity, wall geometry, room faces, and opening attachment.
- Conditional watertightness restricts the denominator to watertight post-edit targets.
- `Verifiability = outside_pixel_exact * outside_room_label_exact * validity * conditional_watertightness`.
- Missing factors produce N/A and are never imputed.

## 5. Graph consistency

- Node F1 and edge overlap use unique room types and room-type-pair adjacency.
- GED uses NetworkX unit node/edge edit costs.
- The canonical instance graph is a separate secondary view.

## 6. Failure taxonomy

- Watertight failure: target is closed but prediction verification fails.
- Node loss/insertion: room-type graph edit path changes node identity/count.
- Edge error: adjacency edge is missed or falsely inserted.
- Room-label drift: semantic identity changes outside the intended edit.

## 7. Validity boundary

- Sparse room counts and weak interfaces can push conditional watertightness below 0.95.
- Strict outside preservation remains separately measured in those bins.
- A global-recomposition edit that legitimately changes the whole plan is outside this contract.

## 8. Sources

- `src/floorflow/evaluation/edit_detection.py`
- `src/floorflow/evaluation/fpeval_bridge.py`
- `third_party/fpvec-lab/fpeval`
- `src/floorflow/evaluation/housemind_structure.py`
- `src/floorflow/evaluation/failure_taxonomy.py`
