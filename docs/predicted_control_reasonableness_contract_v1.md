# Predicted-Control Reasonableness Contract v1

## Decision

The predicted-control track is evaluated for structural reasonableness before
Room IoU optimization. The machine-readable contract is
`configs/predicted_control_reasonableness_contract_v1.json`.

The contract separates four objects that must not be pooled:

- standalone FloorFlow target-free predictions;
- the clearly labeled Raster2Seq-proposal plus FloorFlow-legalizer hybrid;
- official Raster2Seq reconstruction scores;
- oracle and target-guided diagnostics.

## Upstream Alignment

| Metric family | Direct source | Use |
|---|---|---|
| Validity, raw watertightness, wall/opening/room matching, edit cost | `third_party/fpvec-lab` commit `b4b4a169` | Direct dependency through `floorflow.evaluation.fpeval_bridge` |
| Official Room/Corner/Angle/Opening | Raster2Seq commit `a6c4e27a`, `Evaluator.py` and official `result_jsons` | External official score |
| Room-type Node F1, shared-edge overlap, fixed GED | FloorFlow implementation of the HouseMind-style graph view and GSDiff shared-edge protocol | Structure metric with one fixed evaluator |
| One-to-one instance precision/recall/F1 | `room_metric_suite.py` Hungarian matching | Identity diagnostic |
| Plan reasonableness | `generation_plausibility.py` | Target-free promotion gate |

Raster2Seq room matching uses IoU above `0.5` while ignoring the wall-mask
region. Corners use a `10 px` tolerance, angles use `5 degrees`, and opening
endpoints use `10 px` with exact type agreement. The local adapter rasterization
is not an official replacement because Raster2Seq's evaluator depends on its
dataset-specific loader.

HouseMind's public editing metrics are used as protocol context only. Official
HouseMind inference is unavailable, so no direct complete-method reproduction
or leaderboard win is claimed.

## Mandatory Reasonableness Gate

For a predicted-control row to enter fidelity optimization:

- parseability and five-check validity are `1.0`;
- raster plausibility is at least `0.95`;
- room-face retention is at least `0.95`;
- functional watertightness is at least `0.95`;
- wall continuity is at least `0.95`;
- plan plausibility is at least `0.95`;
- room overlap and out-of-boundary ratios are zero;
- raw and componentwise watertightness are reported separately.

The gate is target-free. Target labels are read only by evaluation after
generation. Test503 remains closed.

## 2026 Prediction-Network Route

The first executable route uses the SIGGRAPH 2026 Raster2Seq model as a
target-free proposal teacher:

1. Generate room masks, polygons, classes, counts, and validity on
   `train256` and two disjoint `val64` subsets.
2. Store the predictions as controls, never as ground truth.
3. Distill them into FloorFlow's existing native-raster Mask2Former-style
   `RoomMaskDecoder` with objectness, class, mask, boundary, cardinality, and
   query-diversity losses.
4. Evaluate standalone FloorFlow without Raster2Seq code or weights.

If query distillation fails, the fallback adapts Raster2Seq's anchor-guided
polygon-sequence idea into the existing `PolygonLoopHead`. DPLAN, FloorplanVLM,
and RenoLayout are 2026 design references only until a matched executable is
available.

## Rejected Shortcuts

- Increasing the global wall-gap radius to improve one plausibility counter.
- Relabeling frozen room components or merging by oracle count.
- Treating adapter Room IoU as official Raster2Seq Room IoU.
- Calling the hybrid standalone FloorFlow.
- Ranking different splits or evaluators in one leaderboard.
