# CVPR gap v3 — data/model/metric spine and open-source adapters (2026-10-25)

## What a CVPR submission is still missing

The frozen contour-only target-free test503 result is internally consistent and
passes the legality floors (schema validity 1.0, conditional watertight 0.9865,
room target recall 0.9960, zero outside-footprint pixels). What is still not
defensible as a main result is the reconstruction quality:

| Metric | test503 | Gap driver |
|---|---:|---|
| Room macro IoU | 0.1695 | semantic segmentation + instance assignment |
| Instance geometric IoU | 0.5178 | instance matching, not pixel quality |
| fpvec wall F1 | 0.4195 | wall vectorization readout |
| fpvec opening F1 | 0.1155 | opening heatmap never read out by decoder |
| fpvec room polygon F1 | 0.1608 | room-loop vectorization |
| Furniture IoU | 0.0690 | rare small-object queries under-fit |
| Unconditional watertight | 0.9225 | wall topology closure |

The dominant measured losses, in order, are objects, room segmentation,
vectorization, then topology. The 0.95 hard gate is not identifiable
contour-only (only 0.596% of test503 has a train neighbour at IoU 0.95 and GT
room occupancy peaks at 0.9365), so the paper claim must be framed as a
distributional conditional-generation / conditional-completion result, not
per-case exact reconstruction.

## The organizing spine: model-data link, then level metrics

Every task in `todo.json` is anchored to one hop of the data-to-model link and
one metric level. The levels are:

- L0 legal: schema validity, watertight, room target recall, footprint.
- L1 topology: wall graph F1, opening-wall alignment, containment, gap count.
- L2 objects: wall/opening/furniture IoU and F1.
- L3 instances: instance geometric IoU, type accuracy, room group IoU.
- L4 semantics: room macro IoU, boundary F1, rare-class recall.
- L5 generation: style separation, diversity, runtime.

An improvement is only counted when the owning metric level moves, and every
model module names the level it feeds. This prevents "the number went up but
the paper claim did not".

## Concrete module mapping (do not rebuild, extend)

The plan now points at the exact local symbol instead of inventing new files.
The largest single finding is a wiring gap: `geometry/reconciliation.py::
reconcile_plan` and `geometry/opening_readout.py::read_wall_interval_detections`
already exist and are unit-tested, but `evaluation/fpeval_bridge.py::
fpeval_raster_metrics` still calls `raster_labels_to_wall_plan` directly and
never reads the opening heatmap. That is exactly the fpvec-lab observation
(the decoder emits an opening heatmap it never consumes), so P0 fixes the
wiring before any new training. The full `module_map` with file::symbol
targets lives in `todo.json`.

## Dynamic architecture levers per weak metric

- Furniture IoU (L2): the dynamic Mask2Former-style query budget already
  exists; the remaining loss is query collapse for rare small classes. Lever:
  class-aware dynamic query budgets with rare-class anchor recycling, and a
  differentiable rasterizer (Raster2Seq `diff_ras`) so furniture placement is
  supervised in vector space, not only pixels.
- Room macro IoU (L4): pixel-distribution and semantic-element heads coexist
  but the channel hierarchy is still flat in places. Lever: route the
  pixel-distribution head as the parent gate for room logits, and let the
  semantic-element head receive parent-gated tokens (hierarchical
  cross-attention), so dense pixels and sparse elements do not fight.
- Vectorization (L1/L3): readout is the cheapest large win. Reuse the opening
  heatmap the decoder already emits but never reads (fpvec-lab finding), add
  the FloorplanTransformation integer-programming wall reconciliation, and add
  a polygon-sequence readout head backed by Raster2Seq.
- Topology (L1): low-confidence seam closure exists; pair it with the IP
  junction snapping from FloorplanTransformation.

## Open-source adapters cloned this turn

| Repo | License | Borrow contract |
|---|---|---|
| `art-programmer/FloorplanTransformation` @ `929e7a3` | MIT | Import `pytorch/IP.py` as `src/floorflow/geometry/ip_reconcile.py`; junction/wall snapping runs after wall readout. |
| `Cornell-VAILab/Raster2Seq` @ `a6c4e27` | MIT | Import `diff_ras` for differentiable polygon rasterization in a polygon-sequence head; use as the vector baseline. |
| `SizheHu/GSDiff` @ `a02af02` | GPL-3.0 | Read only; no source copied. |
| `luozn15/FloorplanGAN` @ `92b8fa2` | MIT | Adapt `renderer_g2v` differentiable generator-to-vector rendering for adversarial refinement. |

The precise adaptation plan and acceptance criteria live in `todo.json`
`P0_vector_readout` through `P5_metric_hierarchy`. The reuse-first rule remains:
we vendor a pinned MIT class rather than rewrite the algorithm.
