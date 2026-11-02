# CVPR gap audit (2026-10-25)

## What the current evidence supports

This is a contour-conditioned CAD interior reconstruction study. It is not a
boundary detector: the exterior contour is supplied, and the model predicts
room semantics, interior walls, openings, furniture and room support. The
frozen 503-case result passes the declared legality floors:

- schema validity = 1.0;
- conditional watertightness = 0.9887;
- room target recall = 0.9658;
- outside-footprint pixels = 0.

The implementation also has auditable dynamic architecture: multi-kernel
output routing, hierarchical semantic families, MiTA route attention, dynamic
token budgets and a recursive room graph.

## Why this is not yet a CVPR acceptance argument

The 503-case single-sample target-free result still has:

| Metric | Result |
|---|---:|
| Room macro IoU | 0.1695 |
| Instance geometric IoU | 0.5178 |
| Instance type accuracy | 0.4319 |
| fpvec wall F1 | 0.4195 |
| fpvec opening F1 | 0.1155 |
| fpvec room polygon F1 | 0.1608 |
| Furniture IoU | 0.0690 |
| Unconditional watertight | 0.9225 |

These are not close to a defensible exact-reconstruction claim at 0.95. The
largest measured remaining loss is object geometry, followed by room
segmentation and vectorization.

The gated geometry/semantic decomposition was selected on val64, where it
increased instance geometry from `0.5058` to `0.5270` and balanced CAD from
`0.3199` to `0.3226` without violating the Room-IoU guard. Frozen test503 did
not reproduce that instance gain. This negative transfer is retained in the
paper table and motivates a larger train-only gate fit rather than test-set
selection.

The final model removes train-bank retrieval from the main path and routes a
learned footprint prior through wall, opening, room and furniture channels.
The learned graph trust gate and dynamic furniture queries add only a small
gain at this training budget; opening IoU improves to `0.1884`, but furniture
and vector room geometry remain the dominant gaps.

## Identifiability constraint

On CubiCasa test503, only `0.596%` has an exact contour match in train. Nearest
train-room IoU is `0.1004` on average and only `0.596%` reaches IoU `0.95`.
The GT room pixels occupy at most `0.9365` of the footprint, so the previously
requested `room_coverage >= 0.95` cannot be met even by GT. Exact contour-only
per-case reconstruction at 0.95 therefore conflicts with the dataset and the
information available at inference.

## Two legitimate next contracts

1. Report a distributional conditional-generation claim: sample quality,
   coverage, style control, topology legality and diversity against a
   train-derived reference distribution. Do not label sample means as exact
   reconstruction.
2. Add a legitimate non-target condition available at inference, such as
   wall observations, room counts, a partial CAD sketch, or a user-selected
   program. Then evaluate conditional completion with a contract that does
   not use target-derived controls.

The current model can support the first claim after stronger object and
distributional evidence. It cannot honestly support exact 0.95 reconstruction
under the current target-free contour-only contract.
