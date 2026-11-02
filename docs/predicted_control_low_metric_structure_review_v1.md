# Predicted-control low-metric structure review

## Scope

This audit reviews implementations that directly consume the current low
predicted-control metrics:

- Room macro IoU and instance F1.
- Wall F1 and precision/recall.
- Opening F1.
- Node F1, edge F1, and GED.

The purpose is to separate representational bottlenecks from losses and
decoder errors that can be improved without another blind training run.

## Findings

### 1. Opening loss does not optimize the wall-conditioned consumer

`OpeningDetectionHead` is trained on a dense opening-mask loss, while inference
samples that heatmap on a one-dimensional wall interval. The training objective
never directly rewards predicted probability along reference host walls or
penalizes false intervals on wall background.

Opportunity: add a host-wall sampled Tversky/focal term. Sample target and
negative points along reference wall segments, then optimize the same
probability sequence consumed by `read_wall_interval_detections`.

Priority: P0. This aligns the optimization target with the actual CAD metric.

### 2. Wall-host segmentation loses collinear structure

`raster_labels_to_wall_plan` converts skeleton paths into one segment per
Douglas-Peucker primitive. Opening decoding then treats every segment as an
independent host. Repeated or collinear fragments can split one opening, while
the per-wall cap can duplicate detections on the same physical wall.

The repository already contains collinear consolidation in
`ip_reconcile_plan`, but the opening-host construction does not use it.

Opportunity: create a host-only reconciled wall plan, with no change to final
rooms, and compare interval decoding before and after collinear consolidation.
This is a zero-training experiment.

Status: tested on 12 oracle-host cases. IP reconciliation reduced opening F1
from `0.3004` to `0.2431` and wall count from `27.17` to `23.08`; the current
host does not select the same intervals after consolidation. Keep this as a
negative ablation and move the next budget to wall-band loss.

Priority after the probe: P2.

### 3. Final-plan and opening-host coordinates are mixed

The current end-to-end fallback can detect openings on a high-recall dense wall
plan, then attach them to the higher-precision final plan by nearest-wall index.
The attachment currently has no distance or continuity acceptance rule.

Opportunity: reject or snap an interval only when the projected host is within a
declared distance, and audit validity separately from opening F1. Do not add
off-wall openings merely to improve matching.

Priority: P1.

### 4. Coarse structure decoding computes then discards resolution

`CoarseStructureRefiner` always creates a `256x256` output stage, but the caller
passes the original `128x128` structure-control size as `output_size`. The
model therefore computes the refinement at twice the control resolution and
then downsamples it back before room/wall/opening composition. Thin wall and
opening strokes are specifically vulnerable to this round trip.

Opportunity: retain the 256 output as a separate structural layer or make the
output resolution an explicit training contract. Do not change resolution and
architecture in one unmatched run.

Priority: P1.

### 5. Room queries receive half-resolution features

`RasterInstanceControl` applies `avg_pool2d(..., 2)` before `RoomMaskDecoder`.
Even the decoder's scale-1 feature level is therefore half-resolution. The
query path improves identity and count, but its predicted masks are upsampled
and then hard-routed into class regions by `mask_region_class_logits`.

Opportunity: preserve one full-resolution skip for query-mask logits, and use a
soft per-pixel class/query fusion before the hard safety fallback. Small-room
boundaries and adjacency, which affect Room IoU, Node F1, Edge F1, and GED, are
the expected beneficiaries.

Priority: P0 for graph metrics.

### 6. Hard query-region routing can merge class evidence

`mask_region_class_logits` finds a connected support region and sets that entire
region to one external class. This is safe for identity arbitration but cannot
represent a query whose support contains two true room classes.

Opportunity: compare hard region routing with a learned soft mask over external
per-pixel class probability. Keep the deterministic safety fallback as an
independent guard.

Priority: P1.

### 7. Edge and GED source from room graphs, not wall segments

`housemind_style_metrics` builds room nodes directly from connected room-label
polygons and defines edges by polygon-boundary distance. Wall-readout
improvements alone cannot fix this metric family.

Observed predicted graph size is about `7.42` node types and `13.02` edges versus
`6.06` and `9.95` targets. The dominant issue is extra/duplicated room
adjacency, not missing wall serialization.

Opportunity: treat Node/Edge/GED as room-instance and room-boundary diagnostics.
Improve small-room support, type assignment, fragment merging, and adjacency
precision before attributing graph failures to the wall decoder.

Priority: P0 for a graph-metric claim.

### 8. Component diagnostics use axis-aligned widths

`read_opening_detections` estimates component width from the axis-aligned
bounding box. This is acceptable only as a diagnostic. The main wall-interval
reader already measures width along the host wall and should remain the canonical
consumer.

Priority: P2. Do not tune the component diagnostic and report it as CAD
opening performance.

## Quantified current frontier

- Dedicated opening detector, 256 cases / 24 epochs, oracle-host vector F1:
  `0.2553`.
- Dedicated opening detector, 1,024 cases / 8 epochs, oracle-host vector F1:
  `0.3052`.
- Dedicated opening detector, full 5,126 cases / 8 epochs, oracle-host vector
  F1: `0.3160`.
- Best end-to-end host-mix opening F1: `0.2650` (1,024-case model);
  full-data model: `0.2627`.
- Canonical full-data host-mix wall precision: `0.6233`, Room IoU `0.3325`,
  Edge F1 `0.4906`, GED `9.8281`, validity `1.0`.

The oracle-to-end-to-end gap is therefore at least 0.05 opening F1. It is not
explained by unfolding room depth, safety gates, or graph readout.

## Follow-up objective test

A wall-band weighted focal/Tversky objective was trained under the exact
train1024/cal64, eight-epoch control. It reached oracle-host opening F1
`0.3007` versus `0.3052` for the dense-only objective, so the wall-band
hypothesis did not survive its matched test. Keep the simpler dense detector
and move the next budget to the room-query path that feeds Node/Edge/GED.

## Recommended order

1. Zero-training collinear host-wall consolidation for opening decoding.
2. Host-wall sampled auxiliary opening loss, using a matched 256/1024/5126
   budget ladder.
3. Full-resolution room-query skip plus soft class fusion for graph metrics.
4. Structural-layer resolution contract audit for wall/opening composition.
5. Only after those analyses, change encoder capacity or add another global
   refinement stage.

Every arm must retain the existing room, validity, wall-continuity, and
watertightness non-regression gates.
