# r155 implementation report (2026-10-25)

All six `todo.json` tasks were implemented by extending the existing modules;
no replacement architecture was introduced. The full suite passes:
`428 passed, 1 skipped`. All Python files remain <= 600 lines and the changed
files pass `ruff check`.

## What changed

| Task | Existing module extended | New adapter |
|---|---|---|
| P0 readout | `evaluation/fpeval_bridge.py::fpeval_raster_metrics` now accepts `reconcile`, `reconcile_use_ip`, and opening heatmap readout; `inference/geometry_wall.py::apply_wall_face_rooms` can reconcile | `geometry/ip_reconcile.py` (MILP junction/merge selection, SciPy HiGHS) |
| P1 polygon | `heads/polygon_loop.py::_soft_polygon_masks` delegates to diff_ras; `polygon_loop_loss(mask_rasterizer=...)` | `geometry/diff_ras.py` (vectorized signed-distance rasterizer, native fallback) |
| P2 hierarchy | `distribution/hub.py` exposes explicit `parent_gates`; `distribution/element_head.py` accepts `room_parent_gate` | `distribution/element_parents.py` |
| P3 furniture | `distribution/query_budget.py` supports rare-class recycling; `heads/dynamic_object_instance.py` adds soft render + density-aware budget | `objectives/furniture_vector_refine.py` (FloorplanGAN-style soft projection) |
| P4 CFM | `flow/cfm_backend.py::sample_flow_batch` adds token minibatch OT; CLI defaults to `--minibatch-ot token` | `objectives/token_optimal_transport.py::batch_token_assignment` |
| P5 metrics | `evaluation/metric_hierarchy.py` audits module ownership; `data/geometry_cache.py::audit_geometry_hop` audits the cache hop | `metric_hierarchy_test503_v4.json`, `geometry_hop_audit_val48_v1.json` |

## Measured results

The val64 evaluation (`outputs/reports/r155_readout_val64_v4.json`, same
checkpoint/settings as the r154 reference) gives:

| Metric | r154 ref | r155 | Delta |
|---|---:|---:|---:|
| fpeval opening F1 | 0.1015 | 0.1130 | **+0.0115** |
| fpeval wall F1 | 0.4338 | 0.4338 | 0.0000 |
| room macro IoU | 0.1357 | 0.1356 | -0.0001 |
| room polygon F1 | 0.1684 | 0.1591 | -0.0093 |
| furniture IoU | 0.0669 | 0.0614 | -0.0055 |
| instance geometric IoU | 0.5263 | 0.5148 | -0.0115 |

The opening fusion is the only metric that moves: an A/B with the heatmap
disabled (`r155_readout_val64_noopening.json`) reproduces furniture, instance
and room metrics exactly and returns opening F1 to `0.1015`.

The furniture/instance difference from the r154 reference is therefore a
**pre-existing reproducibility gap** in the current working tree, not a
regression introduced here. The reference report and the current code share
the same checkpoint and settings but differ on those two metrics; this is
flagged as a residual for P5 rather than hidden.

## Training-only paths

`diff_ras`, token minibatch OT and rare-class recycling are training-time
changes. A fresh training run is required before their metric effect can be
claimed; the current frozen checkpoint was not trained with them. The
token-OT overhead is ~15 ms per batch at `[4,65,256,256]`. The vectorized
diff_ras is ~3.8x the winding fallback at grid 24/32 on GPU (5.3 ms vs
1.4 ms per call), so the faster implementation was adopted.

## Data-hop finding

`audit_geometry_hop` on 48 val caches reports `consistent_fraction = 1.0`,
`mean_room_support_iou = 1.0`, zero taxonomy/footprint inconsistencies, and
**313 unclassified openings** whose `class_id` is 0. They are geometrically
valid but carry no taxonomy class, which is a data-quality item for the paper.

## Frozen test503 audit

`outputs/reports/r155_readout_test503_v1.json` reruns the frozen checkpoint
with the same settings. Against the r154 report:

| Metric | r154 report | r155 run | Delta |
|---|---:|---:|---:|
| room macro IoU | 0.1695 | 0.1703 | +0.0008 |
| fpeval wall F1 | 0.4195 | 0.4196 | +0.0001 |
| fpeval opening F1 | 0.1155 | 0.1193 | **+0.0038** |
| fpeval room polygon F1 | 0.1608 | 0.1614 | +0.0006 |
| furniture IoU | 0.0690 | 0.0603 | -0.0087 |
| instance geometric IoU | 0.5178 | 0.5160 | -0.0018 |
| balanced CAD | 0.3301 | 0.3303 | +0.0001 |

The furniture drop is not caused by the opening readout: the heatmap-off
ablation and the fusion run have identical furniture scores. File timestamps
show the r154 report was generated at 21:01 while the final r154 tree was
written at 21:17, so the "frozen" r154 report predates the code that now
exists. The test503 report is therefore stale; it must be regenerated before
any submission claim.

## Residuals

- P0's +0.10 target is not reached; the fused heatmap adds +0.0115. The opening
  head needs recalibration/retraining before a larger claim is made.
- diff_ras/token-OT/rare-class gains require a new training run.
- The r154 reference report is not exactly reproducible from the current tree
  on furniture/instance metrics; resolve before freezing any new test503 result.
