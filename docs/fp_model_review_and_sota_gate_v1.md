# FloorFlow Model Review and Fine-Metric SOTA Gate

Status: historical. The current protocol authority is
`docs/fp_oracle_control_protocol_status_v1.md`; the current claim language is
`paper/cvpr_claim_sections_v3.md`.

## Decision

The strongest defensible CVPR identity is constraint-preserving interactive
floorplan completion with fine-grained edit verification. The strongest public
fine metric is HouseMind's edit-detection IoU (`Delta-IoU`).

Final claim:

> 在 HouseMind 定义的 Delta-IoU 编辑检测协议下，FloorFlow 在 CubiCasa
> test503 上五个 seed 均稳定超过 HouseMind-E 报告的协议锚点 0.608，并一致
> 优于各自 raw renderer；在与公开 Raster2Seq checkpoint 的 matched
> comparison 中，FloorFlow 的 local-delta layer 在相同 2,479 对 pre/post
> rows 和同一 scorer 上的配对 95% CI 排除零，且编辑区域外的像素预测与房间
> 标签保持精确一致。

The retained control-change gated version reaches Delta-IoU
`0.9848-0.9903` and Delta-MSE `0.000168-0.000259` across five seeds. Its
composed topology passes conditional watertightness `0.9579-0.9675`; validity
is `1.0` and outside exactness is `1.0`.

The aggregate `0.874-0.899` and `0.000168-0.000259` values in this historical
note are superseded by the current per-operation range `0.956703-0.999801` and
`0.000019-0.000096`. See the canonical protocol summary for the current claim.

Outside-edit locality is measured as pixelwise exact equality on valid pixels
outside the declared rectangular edit region, plus exact equality of room-label
values. The minimum exact-match ratio is `1.0` for both measurements across all
cases, operations, and seeds.

Predictions are stored as `uint8` categorical CubiCasa labels. Equality is
strict integer equality with no floating-point tolerance. The outside region
is the footprint minus the declared edit rectangle and minus all wall pixels,
including one-pixel dilation of walls in either the pre- or post-edit target.

## Matched External Edit Comparison

We ran the public Raster2Seq CubiCasa5K checkpoint on the identical test503
pre/post edit rows. Both methods receive the same coarse-control information,
and both outputs are scored by the same HouseMind-style evaluator. Raster2Seq
independently regenerates each side and is not edit-aware.

| Operation | Cases | FloorFlow Delta-IoU | Raster2Seq Delta-IoU | Raster2Seq Delta-MSE | Raster2Seq outside exact |
|---|---:|---:|---:|---:|---:|
| room_type | 487 | 0.8353 | 0.1249 | 0.1579 | 0.9545 |
| room_add | 495 | 0.8372 | 0.0991 | 0.1739 | 0.8773 |
| room_remove | 500 | 0.8352 | 0.1649 | 0.1518 | 0.9573 |
| opening_delete | 502 | 0.9616 | 0.0357 | 0.0882 | 0.9178 |
| furniture_delete (supp.) | 495 | 0.9734 | 0.0572 | 0.1188 | 0.8969 |

For every main operation and every FloorFlow seed, the paired Delta-IoU gain
has a 95% bootstrap interval excluding zero. Delta-MSE also favors FloorFlow
relative to Raster2Seq. Furniture deletion is supplementary because Raster2Seq
has no furniture output.

This comparison supports the edit-locality contribution against a strong
CubiCasa reconstruction method. It does not turn Raster2Seq into an
editing-aware baseline, and it does not replace HouseMind's RPLAN benchmark.
The defensible reviewer response is that each edited coarse control induces a
new reconstruction problem; independently reconstructing both sides tests
whether the method responds consistently without access to edit metadata.
Raster2Seq is therefore a matched reconstruction baseline, not a straw-man
editing method.

The outside-edit breakdown shows that `room_add` leaks most often
(`87.73%` exact, `12.27%` changed), followed by supplementary furniture
deletion (`89.69%` exact). Opening deletion is `91.78%` exact, while room type
and room removal average `95.45%` and `95.73%`. The tail remains material:
`room_remove` has a minimum per-case exact ratio of only `0.0232`. FloorFlow
remains `1.0` in every cell by construction.

An editing-aware weak baseline was also evaluated. Raster2Seq was wrapped with
the same edit region and outside freeze, but reached only Delta-IoU `0.1634`
and Delta-MSE `0.0732`. This rules out the explanation that FloorFlow's gain is
entirely due to applying a locality wrapper to an otherwise equivalent model.

Raster2Seq provenance:

- code commit `a6c4e27a68d11d7a459f6e4a2601fd887227dd1a`
- checkpoint SHA-256
  `f7c0ef9379aa8cf11349da8b2179214e0d495af758383984a2361a1309aab3b7`
- official source `https://huggingface.co/haopt/Raster2Seq`
- 4,958 independent pre/post inferences over 2,479 paired edits

Strict SOTA requires one of:

1. Running HouseMind-E and FloorFlow on the same edit set and scorer.
2. Running FloorFlow on HouseMind's 2,309-case edit benchmark.
3. Replacing the anchor with a self-contained public benchmark on which all
   methods are executed.

## Model Review

The retained model is not one end-to-end network. It is four deterministic
refiners:

| Stage | Parameters | Main role | Current risk |
|---|---:|---|---|
| room | 479,928 | local room partition and topology | largest contributor, but control-dependent |
| structure | 501,933 | walls and typed openings | low-level CAD quality remains weak on noisy controls |
| furniture | 506,460 | fixed furniture support and class | dataset covers mostly fixtures, not loose furniture |
| support | 498,657 | room support under occluders | seed-unstable and replaced by room-label support |

The room, structure, and furniture stages beat matched component-off paths.
This is an audited contribution under matched checkpoints, not a general causal
claim about architecture design. MiTA is excluded from the main stack because
its advantage disappears under stronger masking.

The current high clean-control scores are partly guaranteed by the information
in the supplied coarse controls. They support a completion tool, not
contour-only reconstruction SOTA. Predicted controls remain unusable and
missing-control robustness still fails at 10% dropout.

## Strengthening Result

The new `local_delta` renderer composes a new edit result only inside the
declared edit region and preserves the cached pre-edit raster elsewhere. It is
target-free and deterministic. This removes the seed-sensitive outside-edit
topology drift observed in the raw post-edit room renderer.

The HouseMind-style metric is implemented directly:

- `M_gt_delta = target_before != target_after`
- `M_pred_delta = prediction_before != prediction_after`
- `Delta-IoU = |intersection| / |union|`
- `Delta-MSE = mean((M_gt_delta - M_pred_delta)^2)`

Wall pixels are excluded, matching HouseMind appendix B.3.

## Source Matrix

The review uses 21 primary or official sources. “Direct” means a compatible
full-CAD task; “protocol” means a reusable metric or task definition; “context”
means a related method that cannot be used as a matched full-CAD baseline.

| # | Source | Role |
|---:|---|---|
| 1 | [CubiCasa5K](https://doi.org/10.1007/978-3-030-20205-7_3) | primary labeled floorplan data |
| 2 | [HouseDiffusion](https://doi.org/10.1109/CVPR52729.2023.00529) | direct vector floorplan generation baseline |
| 3 | [House-GAN++](https://doi.org/10.1109/CVPR46437.2021.01342) | graph-constrained floorplan baseline |
| 4 | [MaskPLAN](https://doi.org/10.1109/CVPR52733.2024.00856) | partial-input layout generation |
| 5 | [LayoutDM](https://doi.org/10.1109/CVPR52729.2023.01760) | context: generic graphic layout |
| 6 | [HouseMind](https://arxiv.org/abs/2603.11640) | direct unified floorplan generation/editing anchor |
| 7 | [FloorPlanLLaMA](https://doi.org/10.18653/v1/2025.acl-long.331) | language-aligned floorplan generation |
| 8 | [GSDiff](https://doi.org/10.1609/aaai.v39i16.33904) | vector geometry and topology generation |
| 9 | [Raster2Seq](https://doi.org/10.1145/3799902.3811124) | vector polygon sequence generation |
| 10 | [FloorplanMAE](https://arxiv.org/abs/2506.08363) | partial-input floorplan completion |
| 11 | [Boundary-Constrained Diffusion](https://arxiv.org/abs/2602.01949) | boundary-conditioned synthesis |
| 12 | [CG-Floor](https://orca.cardiff.ac.uk/id/eprint/186972/1/CG_Floor_CVPR.pdf) | centroid-guided large-plan generation |
| 13 | [GeoTopoDiff](https://doi.org/10.2312/pg.20261061) | geometry-topology coupling |
| 14 | [TLC-Plan](https://arxiv.org/abs/2602.07100) | two-level codebook vector generation |
| 15 | [DPLAN](https://arxiv.org/abs/2606.21159) | connectivity-focused floorplan generation |
| 16 | [GreenPlanner](https://arxiv.org/abs/2512.00406) | function-feasible practical generation |
| 17 | [ConsistCompose](https://arxiv.org/abs/2511.18333) | context: image composition, not floorplan CAD |
| 18 | [Floorplan Generation Evaluation](https://doi.org/10.1007/978-3-031-87364-5_34) | evaluation protocol |
| 19 | [FloorplanGAN](https://doi.org/10.1016/j.autcon.2022.104470) | vector residential generation |
| 20 | [HouseTune](https://doi.org/10.1609/aaai.v40i16.38417) | LLM-assisted floorplan generation |
| 21 | [Graph Beta Diffusion](https://doi.org/10.1145/3757374.3771518) | graph-based floorplan generation |

`MANSION` was not included because no verifiable paper with that title and a
floorplan-generation interface could be located in Crossref/OpenAlex/GitHub.
It should not be used as a baseline citation until a DOI or official artifact
is supplied.

## Required Next Gates

1. Run HouseMind-E on its released RPLAN edit benchmark when official inference
   code becomes available. The current public release provides data and weights,
   but the project page still marks code as “Code Soon.”
2. Keep the dual-metric result scoped to CubiCasa; do not present it as a
   HouseMind-benchmark leaderboard result.
3. Do not call the local-delta renderer a new generative model; it is a
   constraint-preserving output layer for interactive editing.

## Delta-MSE Diagnosis

Regional decomposition over five seeds shows that the global Delta-MSE is
boundary dominated:

| Region | Area fraction | Delta-MSE mean | Range | Error mass |
|---|---:|---:|---:|---:|
| edit region | 0.1551 | 0.024613 | 0.022962-0.029230 | 0.7359 |
| wall/opening boundary | 0.0411 | 0.039270 | 0.036164-0.048185 | 0.4059 |
| room boundary | 0.0274 | 0.067724 | 0.064121-0.076327 | 0.3372 |
| edit interior | 0.0882 | 0.002032 | 0.001134-0.003809 | 0.0132 |
| outside | 0.8511 | 0.000000 | 0.000000-0.000000 | 0.0000 |

The edit interior is below the HouseMind-E anchor in three of five seeds. This
supports a high-frequency structural-alignment diagnosis, but not a global
Delta-MSE SOTA claim.

A target-free control-change gate was tested as a minimal intervention. It
improved every seed to:

- Delta-IoU `0.9848-0.9903`
- Delta-MSE `0.000168-0.000259`
- outside pixel and room-label exact ratio `1.0`

The first topology audit reported a failure because `per_case.json` rows were
incorrectly paired with sorted raster filenames. Re-pairing by the original
manifest index gives conditional watertightness `0.9579-0.9675`, validity
`1.0`, and no seed regression beyond `0.005`. The earlier rejection is
withdrawn. The candidate therefore passes all five hard gates under the
CubiCasa protocol.

The implementation also includes a target-free guarded topology scaffold that
can fall back to the post-edit topology if a local candidate fails fpeval
verification. The retained five-seed candidate does not require this fallback
for the aggregate gate, but the guard remains available for stricter future
deployments.

The earlier structural appendix used the intermediate topology raster and the
wall-face readout. That combination compressed the target room graph and
reported Node F1 `0.3765-0.3912`, edge overlap `0.3278-0.3407`, and GED
`15.64-16.33`. The corrected canonical protocol applies nearest-room footprint
tiling and connected-polygon extraction symmetrically to prediction and target.
On the final `prediction` output this gives five-seed Node F1
`0.9281-0.9376`, edge overlap `0.8777-0.8978`, and unit GED `1.696-1.969`;
87.5-87.9% of GED cases are exact, with the remainder explicitly marked as
fallback.

Raw wall-continuity percentage is not a valid standalone target: the reference
graph itself has continuity `0.8638`, while a sparser wall output can inflate
the percentage. The corrected target-matched audit gives junction F1
`0.9397-0.9611` and junction-degree exactness `0.9377-0.9716` across five
seeds. Invalid rate, parsable rate, room closure, self-intersection rate, and
out-of-boundary ratio retain their prior legality results.

These canonical structural results remain an oracle-control appendix. They do
not support a deployable room-graph claim or room-graph leaderboard SOTA.

## Claim Guidance

Supported:

> On CubiCasa test503, under HouseMind's Delta-IoU definition, the
> constraint-preserving control-gated local-delta layer reaches Delta-IoU
> 0.9848-0.9903 and Delta-MSE 0.000168-0.000259 across five seeds, preserves
> outside-edit predictions and room labels exactly, maintains validity 1.0 and
> conditional watertightness 0.9579-0.9675, and outperforms a public CubiCasa
> reconstruction method on identical paired edit rows and the shared scorer.

Not supported:

- “FloorFlow beats HouseMind-E on the HouseMind benchmark.”
- “FloorFlow is leaderboard SOTA on HouseMind's Delta-IoU and Delta-MSE.”
- “Raster2Seq was reproduced on HouseMind's benchmark.”

For the unavailable HouseMind leaderboard, the evidence should be defended with
two separate anchors: the published HouseMind-E score and the measured
FloorFlow raw-renderer score. Local-delta must beat both; it may not be described
as a same-benchmark HouseMind result.
