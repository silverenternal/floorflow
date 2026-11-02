# RPLAN Unified Generator Architecture Screen v1

## Outcome

The first end-to-end RPLAN boundary-to-CAD path is implemented and auditable, but
it is not yet a SOTA model. The best 1024/256 screen reaches Room IoU `0.4775`,
instance F1 `0.2391`, type accuracy `0.5806`, and exact room count `0.3906`.
These numbers fail the planned complete-CAD absolute gate and must not be
presented as an external leaderboard result.

## Data Scale

The local CE2EPlan/DiffPlanner preprocessed RPLAN fixture has 56,053 train and
12,018 validation examples. The RPLAN test split remains closed. A 2,000-case
prefix audit found no invalid boundaries, invalid room polygons, positive-area
room overlaps, rooms outside footprints, or malformed adjacency references.

## Contract Audit

The first real-data contract smoke completed:

- room/background/identity/wall/opening/count/adjacency losses produce finite
  gradients;
- generated polygons are assembled into `CanonicalFloorplan`;
- the assembled object passes the vector validator with no errors;
- the object round-trips exactly through the unified token codec.

## Matched Mechanism Screen

At matched 64 train / 32 validation cases, eight epochs, seed 2905, and the same
parameter count:

| Arm | Room IoU | Instance F1 | Type accuracy | Count exact |
| --- | ---: | ---: | ---: | ---: |
| Fixed anchors, separate identity/vector heads | 0.1228 | 0.0333 | 0.1250 | 0.1875 |
| Mask-center anchors, shared identity/vector heads | **0.3161** | **0.0907** | **0.2917** | 0.1875 |

The gain comes from explicit module coupling: `RoomMaskDecoder` predicts identity
masks, their soft centroids initialize vector room polygons, and the same query
class/existence heads condition both identity and polygon losses.

## Data Scaling

| Train / validation | Room IoU | Instance F1 | Type accuracy | Count exact |
| --- | ---: | ---: | ---: | ---: |
| 256 / 64 | 0.3410 | 0.1197 | 0.4010 | 0.5156 |
| 1024 / 256 | **0.4775** | **0.2391** | **0.5806** | 0.3906 |

Four-fold data expansion improves every primary room metric. The 256-case output
was therefore underfit, not a clean representation ceiling.

## Rejected Components

A full-weight dense partition plus dense semantic segmentation auxiliary did not
survive matched 64/32 testing: Room IoU fell from 0.3161 to 0.2374. The code
retains a dense segmentation branch only as a reusable diagnostic/building block;
it is not enabled in the current generator loss.

Removing wall/opening losses because RPLAN lacks interior openings also failed to
improve the running screen. RPLAN's entrance and room-boundary-derived walls stay
in the CAD assembler, while learned wall/opening heads remain available for the
CubiCasa full-CAD stage.

## Bottleneck Diagnosis

An eight-case checkpoint probe reports:

- mean identity mask IoU `0.494`;
- matched identity class accuracy `0.625`;
- rendered room macro IoU `0.355`.

The vector decoder is close to the identity-mask ceiling. The next bottleneck is
room identity, room count, and semantic assignment, not polygon parameterization
or the canonical tokenizer.

## Decision

Retain:

- the target-free RPLAN adapter;
- mask-center room anchors;
- shared identity/vector room heads;
- predicted-count evaluation and checkpoint selection;
- unified CAD assembly and token round-trip;
- CUDA-0 execution with validation-only selection.

Do not retain:

- fixed geometry anchors as the default;
- full-weight dense partition/dense segmentation losses;
- any SOTA claim from the current RPLAN screens.

The next experiment is larger-data scaling with a matched control, followed by
class-balanced and count-aware identity training. The RPLAN test split remains
closed until the validation gate is met.
